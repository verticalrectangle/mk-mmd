"""Projects: a folder with an mk.toml (see docs/design.md: Project file)."""
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

FILENAME = "mk.toml"


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
    build: Path = None
    audio: Path = None
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
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        root = path.parent
        p = data.get("project") or {}
        missing = [k for k in ("fps", "frame0", "duration") if k not in p]
        if missing:
            raise ProjectError(f"{path}: [project] needs {', '.join(missing)}")
        outs = [Output(o["name"], tuple(o["size"])) for o in data.get("output", [])]
        proj = cls(root=root, name=p.get("name", root.name), fps=float(p["fps"]), frame0=int(p["frame0"]),
                   duration=float(p["duration"]), outputs=outs, checks=list(data.get("check", [])), data=data)
        for key in ("blend", "build", "audio"):
            if p.get(key):
                setattr(proj, key, proj.path(p[key]))
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
