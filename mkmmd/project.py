"""Projects: a folder with an mk.toml (see docs/design.md: Project file)."""
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

FILENAME = "mk.toml"
LOOK = "look:"                                              # an include by name: a look of the library
LOOKS = Path(__file__).resolve().parent / "looks"           # mkmmd/looks/<name>.toml


class ProjectError(ValueError):
    pass


@dataclass
class Output:
    name: str
    size: tuple


@dataclass
class Project:
    root: Path
    name: str
    fps: float
    frame0: int
    duration: float
    blend: Path = None
    outputs: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    data: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- time
    def frame(self, t):
        return self.frame0 + t * self.fps

    def time(self, f):
        return (f - self.frame0) / self.fps

    @property
    def n_frames(self):
        return int(round(self.duration * self.fps))

    @property
    def last_frame(self):
        return self.frame0 + self.n_frames - 1

    # ---------------------------------------------------------------- paths
    def path(self, rel):
        p = Path(rel).expanduser()
        return p if p.is_absolute() else self.root / p

    @property
    def mk_dir(self):
        d = self.root / ".mk"
        d.mkdir(exist_ok=True)
        return d

    def output(self, name):
        for o in self.outputs:
            if o.name == name:
                return o
        raise ProjectError(f"no output named {name!r} in {self.root / FILENAME} (have {[o.name for o in self.outputs]})")

    # ---------------------------------------------------------------- cast, colliders, tracks
    @property
    def cast(self):
        return list(self.data.get("cast", []))

    def cast_member(self, name=None):
        """A [[cast]] entry (name, armature, rig, asset). With no name, the only member."""
        cast = self.cast
        if name is None:
            if len(cast) == 1:
                return cast[0]
            raise ProjectError(f"{len(cast)} cast members in {self.root / FILENAME}: say which one (cast = \"...\")")
        for c in cast:
            if c.get("name") == name:
                return c
        raise ProjectError(f"no cast member {name!r} (have {[c.get('name') for c in cast]})")

    def colliders(self, spec):
        """A list of collider specs, inline or by the name of a [colliders] set."""
        if isinstance(spec, str):
            sets = self.data.get("colliders", {})
            if spec not in sets:
                raise ProjectError(f"no collider set {spec!r} in [colliders] (have {sorted(sets)})")
            return list(sets[spec])
        return list(spec or [])

    def track(self, name):
        """tracks/<name>.json: {"frames": [...], "<channel>": [one value per frame], ...}."""
        p = self.root / "tracks" / f"{name}.json"
        if not p.exists():
            raise ProjectError(f"track {name!r} not found ({p})")
        return json.loads(p.read_text(encoding="utf-8"))

    def to_job(self):
        """What the Blender side gets: plain data only."""
        return {"name": self.name, "root": str(self.root), "fps": self.fps, "frame0": self.frame0,
                "duration": self.duration, "blend": str(self.blend) if self.blend else None,
                "outputs": [{"name": o.name, "size": list(o.size)} for o in self.outputs], "data": self.data}

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            path = path / FILENAME
        if not path.exists():
            raise ProjectError(f"{path} not found")
        data = _read(path)
        _clip_grades(data)
        root = path.parent
        p = data.get("project") or {}
        missing = [k for k in ("fps", "frame0", "duration") if k not in p]
        if missing:
            raise ProjectError(f"{path}: [project] needs {', '.join(missing)}")
        outs = [Output(o["name"], tuple(o["size"])) for o in data.get("output", [])]
        proj = cls(root=root, name=p.get("name", root.name), fps=float(p["fps"]), frame0=int(p["frame0"]),
                   duration=float(p["duration"]), outputs=outs, checks=list(data.get("check", [])), data=data)
        if p.get("blend"):
            proj.blend = proj.path(p["blend"])
        return proj

    @classmethod
    def find(cls, start=None):
        """The nearest project at or above `start` (default: the working directory), or None."""
        here = Path(start or Path.cwd()).expanduser().resolve()
        if here.is_file():
            here = here.parent
        for d in [here] + list(here.parents):
            if (d / FILENAME).exists():
                return cls.load(d / FILENAME)
        return None


def _read(path, seen=()):
    """The tables of the project file `path` laid over those of the files its `[project] include` names, each read the
    same way and a later one laid over an earlier one (`_over`)."""
    if path in seen:
        raise ProjectError(f"include loops: {' -> '.join(str(p) for p in (*seen, path))}")
    try:
        with open(path, "rb") as fh:
            own = tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise ProjectError(f"{path}: {e}") from None
    table = own.get("project")
    names = table.get("include") if isinstance(table, dict) else None
    if names is None:
        return own
    names = [names] if isinstance(names, str) else names
    if not isinstance(names, list) or not all(isinstance(n, str) and n.strip() for n in names):
        raise ProjectError(f"{path}: [project] include = {names!r}: expected a file or a look (\"{LOOK}<name>\"), or a "
                           f"list of them")
    data = {}
    for name in names:
        data = _over(data, _read(_included(name, path), (*seen, path)))
    return _over(data, own)


def _over(under, own):
    """The tables `own` laid over the tables `under` (those of an included file): tables merge key by key; a list of
    tables (`[[fx]]`, `[[vector.materials]]`) is joined, `own`'s entries first, so where the first entry that fits
    counts (a material's rule) the nearer file's does; anything else of `own` replaces what was under it."""
    out = dict(under)
    for k, v in own.items():
        was = out.get(k)
        if isinstance(v, dict) and isinstance(was, dict):
            out[k] = _over(was, v)
        elif _tables(v) and _tables(was):
            out[k] = v + was
        else:
            out[k] = v
    return out


def _tables(v):
    return isinstance(v, list) and all(isinstance(x, dict) for x in v)


def _included(name, by):
    """The file an include of the project file `by` names: `look:<name>`, a look of the library, or a path (relative to
    that file's folder; `~` works)."""
    if name.startswith(LOOK):
        p = LOOKS / f"{name[len(LOOK):]}.toml"
        if not p.is_file():
            raise ProjectError(f"{by}: include {name!r}: no such look (have {sorted(q.stem for q in LOOKS.glob('*.toml'))})")
        return p
    p = Path(name).expanduser()
    p = (p if p.is_absolute() else by.parent / p).resolve()
    if not p.is_file():
        raise ProjectError(f"{by}: include {name!r}: {p} not found")
    return p


def _clip_grades(data):
    """A cut-out (`[[fx]] kind = "cutout"`) that names its clip by a registry slug and gives no `grade` takes the grade
    stored with that clip (its entry's `extra.grade`), in place: a hand's skin is corrected once, where the clip is
    registered, and every project shows the same skin. An entry's own `grade` stands as written (`false`: as shot)."""
    fxs = data.get("fx")
    todo = [fx for fx in (fxs if isinstance(fxs, list) else [])
            if isinstance(fx, dict) and fx.get("kind") == "cutout" and "grade" not in fx and isinstance(fx.get("clip"), str)]
    if not todo:
        return
    from . import footage as FT                               # only a project with such a cut-out reads the registry
    from .assets import Registry
    todo = [fx for fx in todo if not FT.is_file(fx["clip"])]
    if not todo:
        return
    stored = {e["slug"]: (e.get("extra") or {}).get("grade") for e in Registry().entries}
    for fx in todo:
        if stored.get(fx["clip"]) is not None:
            fx["grade"] = dict(stored[fx["clip"]])
