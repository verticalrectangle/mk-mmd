"""Asset registry: <assets>/registry.json, one entry per model, motion, prop, audio file, vehicle or reference clip
(docs/design.md: Asset registry). Entries record where an asset lives and what its license asks for; mk never
guesses a license: scanned entries keep the author's readme paths and are marked `unreviewed` until a person (or an
agent that read the readme) fills in license, restrictions and credit."""
import json
import re
from pathlib import Path

from . import config as CFG

KINDS = ("model", "motion", "prop", "vehicle", "audio", "reference", "texture", "font", "other")
FIELDS = ("slug", "kind", "path", "name", "author", "source_url", "license", "restrictions", "credit", "rig", "tags",
          "readmes", "extra")
README_EXT = {".txt", ".md", ".html", ".htm", ".rtf"}
LICENSE_WORDS = re.compile(r"利用規約|規約|禁止|クレジット|著作|配布|改変|license|licence|credit|terms|prohibit|"
                           r"commercial|redistribut|R-18|R18|政治|宗教", re.IGNORECASE)


class AssetError(ValueError):
    pass


def slugify(text):
    s = re.sub(r"[^\w\-]+", "_", str(text), flags=re.UNICODE).strip("_").lower()
    return s[:80] or "asset"


def read_text(path):
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    for enc in ("utf-8-sig", "cp932", "euc_jp"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def find_readmes(path, stop=None):
    """Readme-like text files next to an asset and in its parent folders (up to `stop`)."""
    p = Path(path).resolve()
    here = p.parent if p.is_file() else p
    stop = Path(stop).resolve() if stop else here
    out = []
    for d in [here] + list(here.parents):
        out += sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in README_EXT)
        if d == stop or d == d.parent:
            break
    return out


def license_lines(path, limit=12):
    """Lines of a readme that talk about terms of use (for review; never interpreted)."""
    lines = [ln.strip() for ln in read_text(path).splitlines()]
    return [ln[:200] for ln in lines if ln and LICENSE_WORDS.search(ln)][:limit]


class Registry:
    def __init__(self, root=None):
        self.root = Path(root or CFG.load()["assets"]).expanduser()
        self.path = self.root / "registry.json"
        self.entries = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, slug):
        for e in self.entries:
            if e["slug"] == slug:
                return e
        raise AssetError(f"no asset {slug!r} in {self.path}")

    def find_path(self, path):
        p = str(Path(path).expanduser().resolve())
        return next((e for e in self.entries if e.get("path") == p), None)

    def add(self, entry, replace=False):
        entry = {k: v for k, v in entry.items() if k in FIELDS and v not in (None, "", [])}
        if entry.get("kind") not in KINDS:
            raise AssetError(f"kind must be one of {KINDS}")
        if "path" in entry:
            entry["path"] = str(Path(entry["path"]).expanduser().resolve())
        entry.setdefault("slug", slugify(entry.get("name") or Path(entry.get("path", "asset")).stem))
        for i, e in enumerate(self.entries):
            if e["slug"] == entry["slug"]:
                if not replace:
                    raise AssetError(f"asset {entry['slug']!r} exists (use --replace or `mk assets set`)")
                self.entries[i] = entry
                return entry
        self.entries.append(entry)
        return entry

    def update(self, slug, **fields):
        e = self.get(slug)
        for k, v in fields.items():
            if k not in FIELDS:
                raise AssetError(f"unknown field {k!r} (fields: {', '.join(FIELDS)})")
            if v is None:
                e.pop(k, None)
            else:
                e[k] = v
        return e

    def remove(self, slug):
        e = self.get(slug)
        self.entries.remove(e)
        return e

    def rig_path(self, slug):
        e = self.get(slug)
        if not e.get("rig"):
            raise AssetError(f"asset {slug!r} has no rig.json (run `mk inspect {e.get('path')}` or `mk assets add`)")
        p = Path(e["rig"])
        return p if p.is_absolute() else self.root / p

    def credits(self, slugs):
        """Markdown credits for the given assets, grouped by kind; problems list what is still unreviewed."""
        groups, problems = {}, []
        for s in slugs:
            e = self.get(s)
            line = e.get("credit") or " / ".join(x for x in (e.get("name"), e.get("author")) if x)
            if e.get("source_url"):
                line += f" <{e['source_url']}>"
            groups.setdefault(e["kind"], []).append(line)
            if not e.get("license") or e.get("license") == "unreviewed":
                problems.append(f"{s}: license not reviewed")
            if not e.get("credit") and not e.get("author"):
                problems.append(f"{s}: no author or credit line")
        titles = {"model": "Models", "motion": "Motions", "prop": "Props", "vehicle": "Vehicles", "audio": "Music",
                  "reference": "References", "texture": "Textures", "font": "Fonts", "other": "Other"}
        out = ["# Credits", ""]
        for kind in KINDS:
            if kind in groups:
                out += [f"## {titles[kind]}", ""] + [f"- {ln}" for ln in groups[kind]] + [""]
        return "\n".join(out), problems
