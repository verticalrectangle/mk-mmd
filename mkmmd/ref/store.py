"""Reference sets and their cache layout.

  <cache>/ref/_models/*.task                         MediaPipe model bundles, shared by every set
  <cache>/ref/<project name | default>/<set>/
      clips.json          the set: one entry per remembered clip (small; this is what is worth keeping)
      clips/<id>.mp4      downloads (capped per clip)
      track/<id>.npz      tracking arrays (large)
      contact_sheet.jpg   default output of `mk ref sheet`
      measure.json        default output of `mk ref measure` outside a project

<cache> is ~/.cache/mk (MK_CACHE). Downloads and tracking data are copyrighted material: they stay in the cache and
`clean` deletes them; clips.json keeps ids, authors, urls and license notes so the set can be fetched again."""
import json
import os
import re
import shutil
import time
from pathlib import Path

from .. import config as CFG
from . import RefUsage

DEFAULT_SET = "main"
SCHEMA = 1
_SET_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._") or "default"


def cache_root():
    return CFG.cache_dir() / "ref"


def models_dir():
    return cache_root() / "_models"


def scope_dir(project=None):
    """The cache folder of a project's reference sets (`default` outside a project)."""
    return cache_root() / (slug(project.name) if project is not None else "default")


def set_name(name):
    name = name or DEFAULT_SET
    if not _SET_RE.fullmatch(name):
        raise RefUsage(f"bad set name {name!r}: use letters, digits, '.', '_' or '-' (at most 64 characters)")
    return name


def _tree_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _rm(path):
    """Delete a file or folder; returns the bytes freed."""
    path = Path(path)
    n = _tree_bytes(path)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()
    return n


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class RefSet:
    """One named reference set inside a scope folder."""

    def __init__(self, scope, name=None):
        self.name = set_name(name)
        self.scope = Path(scope)
        self.dir = self.scope / self.name

    # ---------------------------------------------------------------- paths
    @property
    def clips_file(self):
        return self.dir / "clips.json"

    @property
    def video_dir(self):
        return self.dir / "clips"

    @property
    def track_dir(self):
        return self.dir / "track"

    @property
    def sheet_file(self):
        return self.dir / "contact_sheet.jpg"

    @property
    def measure_file(self):
        return self.dir / "measure.json"

    def video_path(self, cid):
        return self.video_dir / f"{int(cid)}.mp4"

    def track_path(self, cid):
        return self.track_dir / f"{int(cid)}.npz"

    # ---------------------------------------------------------------- the set
    def exists(self):
        return self.clips_file.exists()

    def load(self):
        if not self.exists():
            return {"schema": SCHEMA, "set": self.name, "clips": []}
        try:
            data = json.loads(self.clips_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RefUsage(f"{self.clips_file} is not valid JSON: {e}")
        data.setdefault("clips", [])
        return data

    def clips(self):
        return self.load()["clips"]

    def require_clips(self):
        clips = self.clips()
        if not clips:
            raise RefUsage(f"reference set {self.name!r} has no clips ({self.clips_file}): "
                           f"`mk ref search QUERY`, then `mk ref add ID --set {self.name}`")
        return clips

    def put(self, entry, note=None):
        """Add a clip entry or refresh the one with the same id (its `why` survives unless a new note is given).
        Returns 'added' or 'updated'."""
        data = self.load()
        entry = dict(entry)
        entry["added"] = time.strftime("%Y-%m-%d")
        for i, old in enumerate(data["clips"]):
            if old["id"] == entry["id"]:
                entry["why"] = note if note is not None else old.get("why")
                entry["added"] = old.get("added", entry["added"])
                data["clips"][i] = entry
                verb = "updated"
                break
        else:
            entry["why"] = note
            data["clips"].append(entry)
            verb = "added"
        data["schema"], data["set"] = SCHEMA, self.name
        _write_json(self.clips_file, data)
        return verb

    def remove(self, cid):
        data = self.load()
        keep = [c for c in data["clips"] if c["id"] != int(cid)]
        if len(keep) == len(data["clips"]):
            return False
        data["clips"] = keep
        _write_json(self.clips_file, data)
        _rm(self.video_path(cid))
        _rm(self.track_path(cid))
        return True

    # ---------------------------------------------------------------- disk
    def usage(self):
        return {"videos_bytes": _tree_bytes(self.video_dir), "tracks_bytes": _tree_bytes(self.track_dir),
                "other_bytes": _tree_bytes(self.sheet_file)}

    def clean(self, videos=True, tracks=True, forget=False):
        """Delete downloads and/or tracking data (and the contact sheet, which is made of video frames). `forget` also
        deletes the set itself (clips.json, measure.json). Returns what was freed."""
        freed = {}
        if videos:
            freed["videos_bytes"] = _rm(self.video_dir)
            freed["sheet_bytes"] = _rm(self.sheet_file)
        if tracks:
            freed["tracks_bytes"] = _rm(self.track_dir)
        if forget:
            freed["set_bytes"] = _rm(self.dir)
        return freed


def list_sets(scope):
    scope = Path(scope)
    if not scope.exists():
        return []
    return sorted(p.name for p in scope.iterdir() if p.is_dir() and (p / "clips.json").exists())


def clean_scope(scope, videos=True, tracks=True, forget=False, models=True):
    """`clean` for every set under a scope folder, plus the shared model bundles. Returns {set: freed, ...}."""
    out = {}
    for name in list_sets(scope):
        out[name] = RefSet(scope, name).clean(videos, tracks, forget)
    if models:
        out["_models"] = {"models_bytes": _rm(models_dir())}
    return out


def total_freed(report):
    return int(sum(v for r in report.values() for v in r.values()))
