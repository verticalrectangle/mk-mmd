"""Result cache keyed by a hash of the inputs (see docs/design.md: Cache)."""
import ast
import functools
import hashlib
import inspect
import json
import sys
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


def code(module, *names):
    """A fingerprint of the code that makes a result, for a cache key: the top-level functions, classes, methods
    (`"Class.method"`) and assignments `names` of `module`, and everything they use, followed through: the module's
    other top-level names they mention (a method also the methods it calls on `self`), the functions and classes they
    import from the same package, and what they use of the package's modules through an import (`FG.Word`: that name of
    the module; the module passed around alone: all of it). Compared as syntax trees without docstrings, so an edit to
    a comment, a docstring, the layout or code none of it reaches keeps the fingerprint, and any other edit to what it
    reaches changes it. What it cannot see: code reached by name at run time (getattr with a string, globals()), and
    code outside the package (numpy, Blender)."""
    parts = reach(module, *names)
    blob = json.dumps([[m, n, [ast.dump(x) for x in nodes]] for m, n, nodes in parts], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def reach(module, *names):
    """What `code` fingerprints: [(module name, top-level name or None for the whole module, [syntax trees])], sorted."""
    pkg = module.__name__.split(".")[0]
    todo, seen, parts = [(module, n) for n in names], set(), []
    while todo:
        mod, name = todo.pop()
        if (mod.__name__, name) in seen:
            continue
        seen.add((mod.__name__, name))
        tree, defs = _parsed(mod.__file__)
        if name is not None and name not in defs:
            if mod is module and name in names:
                raise KeyError(f"{mod.__name__} has no top-level {name!r}")
            name = None                                 # an imported name made some other way: its module, whole
            if (mod.__name__, None) in seen:
                continue
            seen.add((mod.__name__, None))
        nodes = [tree] if name is None else defs[name]
        cls = name.split(".")[0] if name is not None and "." in name else None
        parts.append((mod.__name__, name, nodes))
        for n in nodes:
            walked = list(ast.walk(n))
            bases = {id(a.value): a.attr for a in walked if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name)}
            for node in walked:
                if cls is not None and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                        and node.value.id == "self" and f"{cls}.{node.attr}" in defs:
                    todo.append((mod, f"{cls}.{node.attr}"))
                if not isinstance(node, ast.Name):
                    continue
                if node.id in defs:
                    if name is not None:
                        todo.append((mod, node.id))
                    continue
                obj = vars(mod).get(node.id)
                if inspect.ismodule(obj):
                    if obj.__name__.split(".")[0] == pkg and getattr(obj, "__file__", None):
                        todo.append((obj, bases.get(id(node))))     # FG.Word: that name of it; FG alone: all of it
                elif inspect.isfunction(obj) or inspect.isclass(obj):
                    src = sys.modules.get(getattr(obj, "__module__", None) or "")
                    if src is mod:                      # defined in a top-level if or try: the module, whole
                        todo.append((mod, None))
                    elif src is not None and src.__name__.split(".")[0] == pkg \
                            and getattr(src, "__file__", None) and "<" not in obj.__qualname__:
                        todo.append((src, obj.__qualname__))
    return sorted(parts, key=lambda p: (p[0], p[1] or ""))


def _parsed(path):
    """(syntax tree without docstrings, {top-level name or "Class.method": [the statements that define it]})."""
    st = Path(path).stat()
    return _parse(str(path), st.st_mtime_ns, st.st_size)


@functools.cache
def _parse(path, mtime_ns, size):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                n.body = b[1:] or [ast.Pass()]
    defs = {}
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.setdefault(st.name, []).append(st)
            if isinstance(st, ast.ClassDef):
                for m in st.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defs.setdefault(f"{st.name}.{m.name}", []).append(m)
        elif isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        defs.setdefault(n.id, []).append(st)
    return tree, defs


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
