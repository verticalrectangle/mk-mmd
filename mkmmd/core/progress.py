"""What a render promises an output's frame folder, and waiting for it (docs/design.md: Post, Following a render):
`mk render` writes MANIFEST into each output's folder before it draws, the files every frame gets (the frame, its
layers, its screen type, its keep mask), and marks it done when it stops, so `mk post --follow` can take each frame as
soon as its files are there. A PNG is there once it ends with its IEND chunk (Blender writes a frame over its claim in
place, mkr renames a finished file into place), a point once it parses.

    write(folder, files, frames)    the manifest: files {frame: [paths relative to folder]} (None: not known, wait for
                                    the end), frames the clip frames the render draws or keeps
    finish(folder, error=None)      mark it done
    finished(path)                  whether a file is all there
    Follow(folder, frames)          .wait(frame): every file the render promised for the frames up to `frame` is there
                                    .ready(frame): whether they are there now
"""
import json
import os
import time
from pathlib import Path

MANIFEST = "render.json"
IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"           # the last chunk of every PNG
POLL = 0.05                                         # s between looks at a file that is not there yet
APPEAR = 60.0                                       # s a follower waits for a manifest to appear


def _save(folder, doc):
    path = Path(folder) / MANIFEST
    tmp = path.with_name(f".{MANIFEST}.{os.getpid()}")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, path)


def write(folder, files, frames, pid=None):
    """The manifest of a render about to draw into `folder`: `files` {frame: [relative paths]} (None when the render
    cannot say which files a frame gets: a follower then waits for the render to end), `frames` the clip frames it
    leaves on disk; `pid` the process that draws (this one)."""
    _save(folder, {"pid": int(pid or os.getpid()), "started": time.time(), "done": False, "error": None,
                   "frames": sorted(int(f) for f in frames),
                   "files": None if files is None else {str(int(f)): list(r) for f, r in sorted(files.items())}})


def read(folder):
    """The manifest in `folder`, or None."""
    try:
        return json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def finish(folder, error=None):
    """Mark the manifest in `folder` done (with `error`, why the render stopped early)."""
    doc = read(folder)
    if doc is not None:
        doc.update(done=True, error=error, ended=time.time())
        _save(folder, doc)


def _alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, TypeError):
        return True
    return True


def over(doc):
    """Whether the render of manifest `doc` has stopped: marked done, or its process is gone."""
    return bool(doc.get("done")) or not _alive(doc.get("pid"))


def finished(path):
    """Whether the file at `path` is all there: not empty, a PNG ending with its IEND chunk, a JSON file that parses."""
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            n = fh.tell()
            if n == 0:
                return False
            if path.suffix == ".png":
                if n < len(IEND):
                    return False
                fh.seek(n - len(IEND))
                return fh.read(len(IEND)) == IEND
            if path.suffix == ".json":
                fh.seek(0)
                json.loads(fh.read().decode("utf-8"))
            return True
    except (OSError, ValueError):
        return False


class Follow:
    """The files a render is writing into `folder`, waited for frame by frame in the order of `frames` (the clip's
    frames). With no manifest there (no render writing, or one that has not started), waits APPEAR s for one, then
    takes the folder as it is."""

    def __init__(self, folder, frames, poll=POLL, appear=APPEAR):
        self.folder = Path(folder)
        self.frames = sorted(int(f) for f in frames)
        self.poll = poll
        self._next = 0                              # every promised file of self.frames[:_next] is there
        self.doc = read(self.folder)
        t0 = time.time()
        while self.doc is None and time.time() - t0 < appear:
            time.sleep(self.poll)
            self.doc = read(self.folder)
        self.waited = 0.0                           # s spent waiting for files

    @property
    def promised(self):
        """The clip frames the render leaves on disk (all of them once it is over), or None without a manifest."""
        return None if self.doc is None else set(self.doc["frames"])

    def _refresh(self):
        got = read(self.folder)
        if got is not None:
            self.doc = got

    def wait(self, frame):
        """Block until every file the render promised for the frames up to `frame` is there; a FileNotFoundError naming
        one when the render stops without it."""
        t0 = time.time()
        self._scan(frame, True)
        self.waited += time.time() - t0

    def ready(self, frame):
        """Whether every file the render promised for the frames up to `frame` is there now (`wait` says what is wrong
        when one never comes)."""
        return self._scan(frame, False)

    def _scan(self, frame, block):
        """Pass over the frames up to `frame` whose promised files are there, with `block` waiting for each; whether
        all of them are."""
        if self.doc is None:
            return True
        while self._next < len(self.frames) and self.frames[self._next] <= frame:
            f = self.frames[self._next]
            files = self.doc.get("files")
            if files is None:                       # not known which: the whole render
                if not block:
                    self._refresh()
                while not over(self.doc):
                    if not block:
                        return False
                    time.sleep(self.poll)
                    self._refresh()
                self._next = len(self.frames)
                break
            for rel in files.get(str(f), ()):
                path = self.folder / rel
                while not finished(path):
                    if not block:
                        return False
                    if over(self.doc):
                        if finished(path):
                            break
                        why = self.doc.get("error") or "it ended"
                        raise FileNotFoundError(f"{path}: the render promised it and stopped without it ({why})")
                    time.sleep(self.poll)
                    self._refresh()
            self._next += 1
        return True
