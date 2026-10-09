"""Character spec: TOML files with `include` lists, merged into one dict.

A spec is one TOML file whose `[model]` table lists the parts to build and the files to merge:

    [model]
    name = "rin"
    parts = ["body", "head", "hair", "outfit"]
    include = ["proportions.toml", "colors.toml", "body.toml"]     # relative to this file; `~` works
    out = "~/mk-assets/models/rin_mk"
    seed = 1

Included files are merged first, in order (later files override earlier ones), then the including file's own content
on top. Tables merge key by key; every other value (lists included) is replaced. Included files may have their own
`[model] include`. The merged dict keeps no `include` key; `Spec.files` lists every file read, in merge order.

Model bases: `include = ["base:girl"]` merges a complete character shipped with mk (`mkmmd/model/bases/girl/model.toml`
and what it includes) under the file's own tables, and a path value `"base:girl/hand.npz"` names a file inside that
base. A character made with `mk model new` is such a file: the base, then only what it changes (docs/model_base.md).

    spec = load("model.toml")
    spec["hair"]["length"]        # a plain dict
    spec.get_path("model.out")    # Path with `~` expanded
    spec.digest()                 # sha1 of the merged content (cache keys)

Unread keys: the builders see the spec through `watch(spec)`, which notes every key they read (also through `merge`
with their defaults); `unread(watched)` lists the rest, so a misspelt or invented key is reported, not silently ignored.
"""
import copy
import hashlib
import json
import tomllib
from pathlib import Path


class SpecError(ValueError):
    pass


class Spec(dict):
    """The merged spec (a dict) plus where it came from: `path` (main file), `dir`, `files` (all files read), `origin`
    (key path (tuple) -> the file that set it last, as text, "--set" for a command-line override; empty for a spec made
    in memory)."""

    path = None
    dir = Path(".")
    files = ()

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.origin = {}

    def section(self, name):
        """A top-level table by name ({} when absent)."""
        v = self.get(name, {})
        return v if isinstance(v, dict) else {}

    def lookup(self, dotted, default=None):
        """`spec.lookup("hair.bangs.length", 0.1)`: nested get with a dotted key."""
        return dig(self, dotted, default)

    def get_path(self, dotted, default=None):
        """A path value with `~` expanded; relative paths resolve against the main file's folder."""
        v = dig(self, dotted, None)
        if v is None:
            return None if default is None else expand(default, self.dir)
        return expand(v, self.dir)

    def digest(self):
        """sha1 of the merged content (stable across key order)."""
        return hashlib.sha1(json.dumps(dict(self), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def dig(d, dotted, default=None):
    """Nested dict lookup with a dotted key; `default` when any step is missing."""
    cur = d
    for k in str(dotted).split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


BASES = Path(__file__).parent / "bases"
BASE_PREFIX = "base:"


def bases():
    """Names of the model bases shipped with mk: the folders of BASES that hold a model.toml."""
    return sorted(p.name for p in BASES.iterdir() if (p / "model.toml").is_file()) if BASES.is_dir() else []


def base_path(ref):
    """`base:NAME` -> that base's model.toml; `base:NAME/sub/path` -> the file at sub/path inside the base."""
    name, _, sub = str(ref)[len(BASE_PREFIX):].partition("/")
    if name not in bases():
        raise SpecError(f"no model base {name!r}; bases: {', '.join(bases()) or 'none'}")
    return BASES / name / (sub or "model.toml")


def expand(path, base=None):
    """Path with `~` expanded; `base:NAME[/file]` names a model base or a file in it (`base_path`); other relative paths
    are taken against `base` (default: the cwd)."""
    if str(path).startswith(BASE_PREFIX):
        return base_path(path)
    p = Path(str(path)).expanduser()
    if not p.is_absolute() and base is not None:
        p = Path(base) / p
    return p


def make_missing(path):
    """Write the missing asset `path` with the maker beside it (make_<stem>.py inside the mkmmd package, whose make(out)
    writes it: a base's make_hand, make_body); False when there is none."""
    import importlib
    p = Path(path)
    maker = p.with_name(f"make_{p.stem}.py")
    root = Path(__file__).resolve().parents[2]                  # the folder that holds the mkmmd package
    if not maker.is_file() or root not in maker.resolve().parents:
        return False
    importlib.import_module(".".join(maker.resolve().relative_to(root).with_suffix("").parts)).make(p)
    return p.is_file()


def merge(a, b):
    """Deep merge: `b` over `a` (None: nothing). Tables merge key by key, everything else in `b` replaces. Returns a new
    dict; the inputs are not changed. With a watched input (a spec table as the builders see it, `watch`) the result is
    watched too: reading a key that came from the spec notes it, and a key the spec tables under it do not set is noted
    as missed there."""
    b = {} if b is None else b
    reads = next((t._reads for t in (b, a) if isinstance(t, Watched)), None)
    srcs = list(dict.fromkeys(s for t in (b, a) if isinstance(t, Watched) for s in t._srcs))     # b's tables win
    out = {} if reads is None else Watched(reads, srcs=srcs)
    for src in (a, b):
        paths = src._paths if isinstance(src, Watched) else {}
        for k, v in dict.items(src):
            cur = dict.get(out, k)
            if src is b and isinstance(v, dict) and isinstance(cur, dict):
                v = merge(cur, v)
            else:
                v = merge(v, None) if isinstance(v, dict) else copy.deepcopy(v)
            dict.__setitem__(out, k, v)
            if reads is not None:
                if k in paths:
                    out._paths[k] = paths[k]
                else:
                    out._paths.pop(k, None)
    return out


# ---------------------------------------------------------------------------------------------------- what was read
class Reads:
    """What the builders read from a watched spec, as paths (tuples of keys): `seen` the keys whose values were read,
    `missed` the keys looked up that the spec does not set (a builder would read them once set), `listed` the tables
    whose keys were listed (their key set matters)."""

    def __init__(self):
        self.seen, self.missed, self.listed = set(), set(), set()

    def snapshot(self):
        return set(self.seen), set(self.missed), set(self.listed)

    def since(self, snap):
        """(seen, missed, listed) added since `snapshot()` returned `snap`."""
        return self.seen - snap[0], self.missed - snap[1], self.listed - snap[2]

    def add(self, seen, missed, listed):
        self.seen |= set(seen)
        self.missed |= set(missed)
        self.listed |= set(listed)


class Watched(dict):
    """A spec table that notes what is read from it (`Reads`): looking a key up ([k], get, in, setdefault, pop) notes
    its value as seen, or as missed where the spec does not set it; listing or copying the table (iteration, keys,
    values, items, copy, dict(t), {**t}, update from it) notes every value and the table as listed. Its tables are Watched
    too. `_paths`: key -> the spec path of its value (keys a builder's defaults gave, merged in with `merge`, have none);
    `_srcs`: the spec paths of the tables it stands for, the one that wins first."""

    def __init__(self, reads, paths=None, srcs=()):
        super().__init__()
        self._reads = reads
        self._paths = {} if paths is None else paths
        self._srcs = tuple(srcs)

    def _note(self, k):
        p = self._paths.get(k)
        if p is not None:
            self._reads.seen.add(p)
        for s in self._srcs:                        # the tables above the one that set it (all: none set it)
            q = s + (k,)
            if q == p:
                break
            self._reads.missed.add(q)

    def _note_all(self):
        self._reads.seen.update(self._paths.values())
        self._reads.listed.update(self._srcs)

    def __getitem__(self, k):
        self._note(k)
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        self._note(k)
        return dict.get(self, k, default)

    def __contains__(self, k):
        self._note(k)
        return dict.__contains__(self, k)

    def setdefault(self, k, default=None):
        self._note(k)
        return dict.setdefault(self, k, default)

    def pop(self, k, *default):
        self._note(k)
        return dict.pop(self, k, *default)

    def __iter__(self):
        self._note_all()
        return dict.__iter__(self)

    def keys(self):
        self._note_all()
        return dict.keys(self)

    def values(self):
        self._note_all()
        return dict.values(self)

    def items(self):
        self._note_all()
        return dict.items(self)

    def copy(self):
        self._note_all()
        return dict(dict.items(self))

    def __deepcopy__(self, memo):
        return merge(self, None)                    # still watched: reads of the copy are noted like the original's


class WatchedSpec(Watched, Spec):
    """`watch(spec)`: the whole spec, watched, with the Spec attributes and methods."""


def watch(spec):
    """A copy of `spec` (its tables; the values are shared) that notes what is read from it (`_reads`); `unread`
    reports the keys nobody read."""
    out = WatchedSpec(Reads(), srcs=((),))
    out.path, out.dir, out.files = getattr(spec, "path", None), getattr(spec, "dir", Path(".")), getattr(spec, "files", ())
    out.origin = getattr(spec, "origin", {})

    def fill(dst, src, prefix):
        for k, v in dict.items(src):
            dst._paths[k] = prefix + (k,)
            if isinstance(v, dict):
                child = Watched(dst._reads, srcs=(prefix + (k,),))
                fill(child, v, prefix + (k,))
                v = child
            dict.__setitem__(dst, k, v)
    fill(out, spec, ())
    return out


def at(spec, path):
    """(True, value) at `path` in `spec`, (False, None) when it is not set; looks without noting a read."""
    cur = spec
    for k in path:
        if not isinstance(cur, dict) or not dict.__contains__(cur, k):
            return False, None
        cur = dict.__getitem__(cur, k)
    return True, cur


def leaves(d, prefix=()):
    """The paths (tuples) of every value in `d` that is not a table, tables entered."""
    for k, v in dict.items(d):
        if isinstance(v, dict):
            yield from leaves(v, prefix + (k,))
        else:
            yield prefix + (k,)


def dotted(path):
    """A spec path (tuple of keys) as text: keys joined by dots, a key with a dot in it quoted."""
    return ".".join(f'"{k}"' if "." in str(k) else str(k) for k in path)


def from_base(origin):
    """Whether a key's origin (`Spec.origin` value) is a file of a model base shipped with mk."""
    return origin not in (None, "--set") and Path(origin).is_relative_to(BASES.resolve())


def unread(spec, tables=None, bases=False):
    """What no builder read from a watched spec (`watch`), in spec order: [(path, n)], path a tuple of keys, n None for
    a key, else the number of keys under a table nothing in was read (reported once as a whole; empty tables are not, and
    a table made only of tables, like [colors], is never whole: its tables are reported). Only the author's keys count:
    those a model base set are left out (the bases keep keys for features a character may switch on) unless `bases`.
    `tables`: only these top-level tables (default: all but [model], which the command line reads)."""
    seen = spec._reads.seen
    touched = {p[:i] for p in seen for i in range(1, len(p))}           # tables something below was read from
    origin = getattr(spec, "origin", {})
    out = []

    def mine(p):
        return bases or not from_base(origin.get(p))

    def walk(t, top):
        for k, v in dict.items(t):
            if top and (k == "model" or (tables is not None and k not in tables)):
                continue
            p = t._paths[k]
            if not isinstance(v, Watched):
                if p not in seen and mine(p):
                    out.append((p, None))
            elif p in touched or (len(v) and all(isinstance(x, dict) for x in dict.values(v))):
                walk(v, False)
            else:
                n = sum(1 for q in leaves(v, p) if mine(q))
                if n:
                    out.append((p, n))
    walk(spec, True)
    return out


def _read(path):
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise SpecError(f"spec file {path} does not exist") from None
    except tomllib.TOMLDecodeError as e:
        raise SpecError(f"{path}: {e}") from None


def _load(path, seen, files, origin):
    path = Path(path).expanduser().resolve()
    if path in seen:
        chain = " -> ".join(str(p) for p in [*seen, path])
        raise SpecError(f"include cycle: {chain}")
    own = _read(path)
    model = own.get("model")
    incs = []
    if isinstance(model, dict) and "include" in model:
        incs = model["include"]
        if isinstance(incs, str):
            incs = [incs]
        if not isinstance(incs, list) or not all(isinstance(i, str) for i in incs):
            raise SpecError(f"{path}: [model] include must be a list of paths")
        model = dict(model)
        del model["include"]
        own = {**own, "model": model}
    merged = {}
    for inc in incs:
        merged = merge(merged, _load(expand(inc, path.parent), (*seen, path), files, origin))
    merged = merge(merged, own)
    origin.update((p, str(path)) for p in leaves(own))         # in merge order: the file that set a key last wins
    files.append(path)
    return merged


def parse_value(text):
    """A command-line value as TOML (`0.3`, `true`, `"a"`, `[1, 2]`); falls back to the raw string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_overrides(spec, overrides):
    """Set dotted keys from `["hair.back.end_above_chin=-0.05", ...]` (CLI `--set`); returns the spec."""
    for item in overrides or ():
        if "=" not in item:
            raise SpecError(f"override {item!r}: expected key=value")
        key, val = item.split("=", 1)
        keys = key.strip().split(".")
        cur = spec
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
            if not isinstance(cur, dict):
                raise SpecError(f"override {item!r}: {k} is not a table")
        cur[keys[-1]] = parse_value(val.strip())
        if isinstance(getattr(spec, "origin", None), dict):
            spec.origin.update((p, "--set") for p in leaves({keys[-1]: cur[keys[-1]]}, tuple(keys[:-1])))
    return spec


def load(path, overrides=None):
    """Load and merge a spec file (see the module docstring); `base:NAME` loads a model base. `overrides`: ["a.b=1", ...]
    applied last."""
    path = expand(path)
    files, origin = [], {}
    merged = _load(path, (), files, origin)
    spec = Spec(merged)
    spec.path = Path(path).expanduser().resolve()
    spec.dir = spec.path.parent
    spec.files = tuple(files)
    spec.origin = origin
    apply_overrides(spec, overrides)
    return spec


def from_dict(data, base=None):
    """A Spec from an in-memory dict (tests, scripts); `base` is the folder relative paths resolve against."""
    spec = Spec(copy.deepcopy(data))
    spec.dir = Path(base) if base else Path(".")
    return spec


def model_cfg(spec):
    """The `[model]` table with defaults: name, parts (required), seed, out (a Path), scale, builders."""
    m = dict(spec.get("model", {}))
    if not isinstance(m.get("parts"), list) or not m["parts"]:
        raise SpecError("[model] parts must list the parts to build, e.g. parts = [\"body\", \"head\"]")
    base = getattr(spec, "dir", None)
    name = str(m.get("name", "model"))
    return {
        "name": name,
        "parts": [str(p) for p in m["parts"]],
        "seed": int(m.get("seed", 1)),
        "out": expand(m.get("out", f"~/mk-assets/models/{name}"), base),
        "scale": float(m.get("scale", 0.08)),
        "builders": dict(m.get("builders", {})),
        "needs": {k: list(v) for k, v in dict(m.get("needs", {})).items()},
        "comment": str(m.get("comment", "")),
    }
