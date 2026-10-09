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
"""
import copy
import hashlib
import json
import tomllib
from pathlib import Path


class SpecError(ValueError):
    pass


class Spec(dict):
    """The merged spec (a dict) plus where it came from: `path` (main file), `dir`, `files` (all files read)."""

    path = None
    dir = Path(".")
    files = ()

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


def merge(a, b):
    """Deep merge: `b` over `a`. Dicts merge key by key, everything else in `b` replaces. Returns a new dict."""
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path):
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise SpecError(f"spec file {path} does not exist") from None
    except tomllib.TOMLDecodeError as e:
        raise SpecError(f"{path}: {e}") from None


def _load(path, seen, files):
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
        merged = merge(merged, _load(expand(inc, path.parent), (*seen, path), files))
    merged = merge(merged, own)
    files.append(path)
    return merged


def parse_value(text):
    """A command-line value as TOML (`0.3`, `true`, `"a"`, `[1, 2]`); falls back to the raw string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_overrides(spec, overrides):
    """Set dotted keys from `["hair.length=0.3", ...]` (CLI `--set`); returns the spec."""
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
    return spec


def load(path, overrides=None):
    """Load and merge a spec file (see the module docstring); `base:NAME` loads a model base. `overrides`: ["a.b=1", ...]
    applied last."""
    path = expand(path)
    files = []
    merged = _load(path, (), files)
    spec = Spec(merged)
    spec.path = Path(path).expanduser().resolve()
    spec.dir = spec.path.parent
    spec.files = tuple(files)
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
