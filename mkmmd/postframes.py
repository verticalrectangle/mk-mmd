"""The frames of one output as `mk post` encodes them (docs/design.md: Post): the cut's frame and the layers mk render drew
beside it, the cut effects (mkmmd.cutfx), the frame effects (mkmmd.framefx), the lens streaks, the screen type, the loop's
wrap and the grade (mkmmd.post), finished in frame order by this process or by worker processes over chunks of
consecutive frames, and handed on in order; following a running render (mkmmd.core.progress), each frame once the
render has its files.

    Output(proj, plan, name, src, files, frames, size, keep)   one output's pipeline; .finish(k) finishes a frame here,
                                     .plan(k) hands what mkr-post can do of it on (with `gpu`, mkmmd.postgpu)
    jobs(n, size)                    how many processes finish `n` frames of `size` (w, h) on this machine by default
    finished(out, jobs, server=None) every frame of `out`, in order, by `jobs` processes and mkr-post: (k, rgb24, stats)

A frame depends on the files on disk and the frames before it only where this module re-creates them, so any split
into chunks gives the frames one pass in order gives:
  - a frame missing on disk (--allow-gaps) repeats the last frame of the clip rendered before it, with its screen type
    and keep mask (before the first: the first frame rendered, with neither), read again where a chunk needs it;
  - the frames of the loop's wrap turn over frame 0 as finished, which a process that has not finished it finishes
    first with effects of its own that have drawn nothing before, as the pass in order does;
  - a chunk starts only on a frame no effect holds a picture of the frames before for (mkmmd.framefx.FrameFx.safe: a
    film's frame over its step, a booth's captures, an eraser's frame, a flowing mosh); a mosh without a flow works its
    picture out again from the cut's frames, and every other memo of the effects (decoded frames, coordinate grids,
    sprites, a clip's decoded frames) is a function of what it is keyed by."""
import multiprocessing
import os
import queue
import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from . import cutfx as CF
from . import footage as FT
from . import framefx as FF
from . import gpufx as GFX
from . import mandala as MDF
from . import memory as MEM
from . import pageturn as PAGE
from . import picturefx as PFX
from . import post as P
from . import postgpu as PG
from .core import framefx as FX
from .core import keep as KP
from .core import palette as PAL
from .core import screentype as SRT
from .core import transition as TR

CHUNK = 12                  # consecutive frames a worker finishes in one go (an effect running on frame to frame goes on)
AHEAD = 2                   # chunks in flight per worker, so the encoder seldom waits for one to start
LOOKAHEAD = 8               # frames handed to mkr-post before the first of them is taken back
SINK_LOOKAHEAD = 32         # ... when mkr-post writes them into the encoder itself: planning runs on that far ahead
INLINE_WORK = 24 * 1920 * 1080   # pixels x frames under which one process finishes the frames (workers take seconds to start)
MAX_JOBS = 4                # workers by default: the machine is shared (renders, builds) and its memory runs out first
GPU_JOBS = 3                # ... when mkr-post finishes the frames: the workers only plan them, ahead of the GPU
WORKER_BYTES = 1.4e9        # a worker's peak at 1920 x 1080 (its effects' pictures and memos), scaled by pixels
WORKER_BASE = 0.3e9         # and whatever the frame's size: the interpreter and its modules
MEMORY_SHARE = 0.5          # of the memory available, what the workers may take between them
EARLY_BYTES = 512 << 20     # heavy frames' pictures (w x h x 12 bytes) planned ahead of their turn and not yet handed on
SEND_WAIT = 30.0            # s the frames' sender gets to stop once its jobs are no longer wanted


class SetupError(ValueError):
    """What is wrong with an output before any frame is finished, as the user should read it."""


def plates(plan, src, files, size):
    """callable(shot, frame) -> what `shot` shows at `frame` (a collage's own `shot`): the cut's frame where it is the
    shot in the cut, else its plate beside the frames in `src`; None when that is not there."""
    def get(shot, g):
        if TR.in_force(plan, shot, g):
            return P.read(files[g], size) if g in files else None
        return P.read(src / TR.plate_rel(shot, g), size)
    return get


def settled_by(plan):
    """The last frame whose layers mkmmd.cutfx.CutFx.check reads to settle the cut effects (an expand's, a collapse's
    or a slice's window), or None."""
    return max((tr["last"] for tr in (plan or {}).get("transitions") or ()
                if tr["kind"] in ("expand", "collapse", "slice")), default=None)


class Output:
    """One output's frames as mk post finishes them. `plan` is the cut's plan (mkmmd.core.transition.plan, None without
    effects), `src` the output's frame folder, `files` {Blender frame: PNG} of the frames rendered there, `frames` the
    clip's Blender frames in order, `size` (w, h) the pictures are graded at and `keep` the `[post] keep` table
    (mkmmd.core.keep.spec) or None. With `verify` the layers the cut effects need are checked first (missing ones are an
    error unless `allow_gaps`) and what they settle on the pictures is settled. With `gpu` (a folder for the pictures
    handed on) `plan` hands what mkr-post can do of a frame to it (mkmmd.postgpu). With `follow` (a
    mkmmd.core.progress.Follow: `files` holds the frames a running render promised) each frame is taken once the render
    has its files (`wait`), and what the cut effects settle once it has their windows. A SetupError says what is wrong."""

    def __init__(self, proj, plan, name, src, files, frames, size, keep=None, verify=False, allow_gaps=False, gpu=None,
                 follow=None):
        self.args = {"proj": proj, "plan": plan, "name": name, "src": src, "files": files, "frames": frames,
                     "size": size, "keep": keep, "gpu": gpu}   # what a worker process builds its own from
        self.gpu = None if gpu is None else Path(gpu)
        self.follow = follow
        post = proj.data.get("post", {})
        look = proj.data.get("look", {})
        pal = PAL.get(look.get("palette", "rose-pine-moon"), look.get("slots"))
        self.name, self.src, self.files, self.frames = name, Path(src), dict(files), list(frames)
        self.size, self.keep = (int(size[0]), int(size[1])), keep
        self.screen = post.get("screen", "graded")
        if self.screen not in ("graded", "over"):
            raise SetupError(f"[post] screen = {self.screen!r}: \"graded\" (the grade covers the screen type) or \"over\"")
        self.grade = P.Grade(post, pal, self.size)
        st = post.get("streaks")
        try:
            self.streaks = P.Streaks({} if st is True else dict(st), pal, self.size) if st else None
        except (ValueError, KeyError) as e:
            raise SetupError(str(e).strip('"')) from None
        footage = FT.source(proj.root)
        self.fx = CF.CutFx(plan, self.src, self.size, name, footage_of=footage, sprite_of=MDF.sprite_source(proj.root),
                           hits_of=lambda: FF.project_hits(proj.root, proj.data)) if TR.has_effects(plan) else None
        if self.fx and verify:
            if follow is None:
                lacking = self.fx.missing(self.frames)
            else:                                          # the render's: the windows CutFx.check reads, all there
                lacking = []
                last = settled_by(plan)
                try:
                    if last is not None:
                        follow.wait(last)
                except FileNotFoundError as e:
                    raise SetupError(f"{name}: {e}") from None
            if lacking and not allow_gaps:
                raise SetupError(f"{name}: {len(lacking)} layer files of transitions / inserts / fx missing in {self.src} "
                                 f"(first {lacking[:3]}); mk render draws them, or pass --no-transitions or --allow-gaps")
            if not lacking:
                try:
                    self.fx.check(self.frames)
                except TR.TransitionError as e:
                    raise SetupError(str(e)) from None
        self.ffx = None
        if (plan or {}).get("fx"):
            files, src, sz = self.files, self.src, self.size
            try:
                self.ffx = FF.FrameFx(plan["fx"], sz, fps=proj.fps, frame0=proj.frame0,
                                      stem_of=FF.stem_source(proj.root, proj.mk_dir / "cache" / "timeline"),
                                      hits=FF.project_hits(proj.root, proj.data), output=name, footage_of=footage,
                                      font_of=PFX.font_source(proj.root), sprite_of=MDF.sprite_source(proj.root),
                                      frame_of=lambda g: P.read(files[g], sz) if g in files else None,
                                      point_of=lambda key, g: CF.read_point(src, key, g, sz),
                                      figure_of=lambda key, g: CF.read_figure(src, key, g, sz),
                                      plate_of=plates(plan, src, files, sz), preload=verify,
                                      paths=GFX.Paths(files, src, sz, plan))
            except FX.FxError as e:
                raise SetupError(str(e)) from None
        self.wrap = TR.wrap_of(plan)                       # the loop's wrap; frame 0 as finished, under its page
        self.lead = P.read(self.files[min(self.files)], self.size)   # stands in for frames before the first one rendered
        self._held = None
        self._carried = (None, None)                       # (frame, its inputs) last repeated over a gap

    @property
    def notes(self):
        """What the cut effects settled on the pictures (the scales expands and collapses reach), for the report."""
        return list(self.fx.notes) if self.fx else []

    def safe(self, k):
        """Whether a process that starts on the clip's frame at index `k` finishes it as the pass through every frame
        before it does (mkmmd.framefx.FrameFx.safe): a worker's chunk of frames starts on such a frame."""
        return self.ffx is None or self.ffx.safe(self.frames[k], self.files)

    def heavy(self, k):
        """Whether planning the clip's frame at index `k` for mkr-post works its picture out here (a pixel sort's
        entries before it: mkmmd.postgpu.first), some fifty times a plain frame's work: such a frame is a worker's
        chunk of its own (chunks)."""
        if self.gpu is None or self.ffx is None:
            return False
        entries = FX.at(self.ffx.fxs, self.frames[k])
        return any(getattr(GFX.KINDS.get(e["kind"]), "FIRST", False) for e in entries[1:])

    def wait(self, k):
        """Following a render: block until it has every file of the clip's frames up to the one at index `k` (a
        FileNotFoundError when it stops without one); nothing to wait for otherwise."""
        if self.follow is not None:
            self.follow.wait(self.frames[k])

    def ready(self, k):
        """Following a render: whether it has every file of the clip's frames up to the one at index `k` now (`wait`
        says what is wrong when one never comes); always otherwise."""
        return self.follow is None or self.follow.ready(self.frames[k])

    # ---------------------------------------------------------------- one frame
    def _read(self, f):
        """(picture, screen layer, keep mask) of a rendered frame `f`."""
        layer = P.read_rgba(self.src / SRT.layer_rel(f), self.size)
        km = P.read(self.src / KP.rel(f), self.size) if self.keep else None
        return P.read(self.files[f], self.size), layer, km

    def _inputs(self, f):
        """(picture, screen layer, keep mask) frame `f` starts from: its own, or over a gap the last frame of the clip
        rendered before it (the first frame rendered, with no layer and no mask, before any)."""
        if f in self.files:
            return self._read(f)
        g = max((h for h in self.frames if h < f and h in self.files), default=None)
        if g is None:
            return self.lead, None, None
        if self._carried[0] != g:
            self._carried = (g, self._read(g))
        return self._carried[1]

    def _before_grade(self, f):
        """Frame `f` as finished before the wrap and the grade: (picture, screen type still to lay over the graded
        picture, keep mask, screen type frames, effects that found a layer missing)."""
        raw, layer, km = self._inputs(f)
        fx, ffx = self.fx, self.ffx
        mask = None if km is None or (fx and fx.active(f)) or (ffx and ffx.active(f)) else km[..., 0]
        img, bare, typed = raw, 0, 0
        if fx and fx.active(f):
            try:
                img = fx.frame(f, raw)
            except FileNotFoundError:                             # layers missing (allowed): the plain cut shows
                bare += 1
        if ffx and ffx.active(f):
            try:
                img = ffx.frame(f, img)
            except (FT.FootageError, FX.FxError):                 # a clip that fails, a pivot with no figure: an error
                raise
            except FileNotFoundError:                             # a point or a frame missing (allowed): left out there
                bare += 1
        if self.streaks is not None:                              # lens streaks: from the picture's lights, not the type
            img = self.streaks(img)
        if layer is not None and self.screen == "graded":        # screen type: over the cut and what its effects made of it
            img = SRT.composite(img, layer)
            typed += 1
            if mask is not None:                                  # the type over a kept eye is graded as type
                mask = mask * (1.0 - layer[..., 3])
        over = layer if self.screen == "over" else None           # or over the finished picture, the grade under it
        return img, over, mask, typed, bare

    def finish(self, k):
        """The clip's frame at index `k` finished: (rgb24 bytes, (darkest luma, screen type frames, kept, effects that
        found a layer missing))."""
        f = self.frames[k]
        img, over, mask, typed, bare = self._before_grade(f)
        w = self.wrap
        if w is not None:                                         # the loop's wrap: this frame as finished turned over
            if f == w["still"]:                                   # frame 0 as finished, held still
                self._held = (img, over, mask)
            if w["first"] <= f <= w["last"]:
                if self._held is None:                            # frame 0 as the pass in order finishes it: first,
                    self._held = Output(**self.args)._before_grade(w["still"])[:3]     # with nothing drawn before
                img, over, mask = PAGE.wrap(w, f, (img, over, mask), self._held)
        g = self.grade(img, k, keep=mask)
        kept = mask is not None
        if over is not None:
            g = SRT.composite(g, over)
            typed += 1
        return P.to_u8(g).tobytes(), (P.luma_min(g), typed, int(kept), bare)

    def plan(self, k):
        """The clip's frame at index `k` for mkr-post (with `gpu`): ("gpu", job, stats) when it can finish the frame
        (mkmmd.postgpu: what the cut effects and the entries before the last it cannot draw make of the frame is worked
        out here and handed on as a picture), ("cpu", rgb24 bytes, stats) finished here otherwise (the loop's wrap, a
        project without `gpu`). stats as finish's, the darkest luma None for mkr-post to say."""
        f = self.frames[k]
        w = self.wrap
        if self.gpu is None or (w is not None and (f == w["still"] or w["first"] <= f <= w["last"])):
            return ("cpu", *self.finish(k))
        g = f if f in self.files else max((h for h in self.frames if h < f and h in self.files), default=None)
        fx, ffx = self.fx, self.ffx
        if ffx is not None:
            ffx.paths.scratch = self.gpu                          # where sticker tiles are written for mkr-post
        entries = FX.at(ffx.fxs, f) if ffx else []
        img, bare, ops = None, 0, []
        if fx and fx.active(f):
            img = self._inputs(f)[0]
            try:
                img = fx.frame(f, img)
            except FileNotFoundError:                             # layers missing (allowed): the plain cut shows
                bare += 1
        if entries:
            before = img
            try:
                cut, tail = len(entries), []                      # entries[cut:] are mkr-post's, tail their ops
                while cut:
                    got = PG.ops(ffx, entries[cut - 1], f)
                    if got is None:
                        break
                    tail.append(got)
                    cut -= 1
                    if PG.first(entries[cut], got):               # a pixel sort: what comes before it is drawn here
                        break
                for e in entries[:cut]:
                    img = ffx._apply(e, f, self._inputs(f)[0] if img is None else img)
                ops = [op for got in reversed(tail) for op in got]
            except (FT.FootageError, FX.FxError):                 # a clip that fails, a pivot with no figure: an error
                raise
            except FileNotFoundError:                             # a point or a frame missing (allowed): left out there
                img, ops, bare = before, [], bare + 1
                PG.dropped(ffx, f)
        job = {"k": k, "ops": ops}
        if img is None:
            job["png"] = str(self.files[g] if g is not None else self.files[min(self.files)])
        else:
            job["raw"] = str(self.gpu / f"{os.getpid()}-{k}.f32")
            np.ascontiguousarray(img, np.float32).tofile(job["raw"])
        layer = None if g is None else self.src / SRT.layer_rel(g)
        if layer is not None and layer.exists():
            job["layer"] = str(layer)
        mask = None if g is None or not self.keep or (fx and fx.active(f)) or entries else self.src / KP.rel(g)
        if mask is not None and mask.exists():
            job["keep"] = str(mask)
        return "gpu", job, (None, int("layer" in job), int("keep" in job), bare)


# ---------------------------------------------------------------- over processes
def jobs(n, size):
    """How many processes finish `n` frames of `size` (w, h) by default: one for a little work (workers take seconds to
    start), else MAX_JOBS, no more than the CPU threads but one, and no more than MEMORY_SHARE of the memory available
    holds at a worker's peak (WORKER_BASE + WORKER_BYTES at full HD, by pixels); unknown memory: one."""
    w, h = int(size[0]), int(size[1])
    if n * w * h < INLINE_WORK:
        return 1
    cpus = max(1, (os.cpu_count() or 2) - 1)
    per = WORKER_BASE + WORKER_BYTES * (w * h) / (1920 * 1080)
    avail = MEM.available()
    if avail is None:
        return 1
    return max(1, min(MAX_JOBS, cpus, int(avail * MEMORY_SHARE // per)))


class Planners:
    """Worker processes that plan chunks of an output's frames (Output.plan), made once for a run and shared by its
    outputs, which go one after another: a worker builds an output's Output the first time it gets a chunk of it (and
    lets the one before go)."""

    def __init__(self, n):
        self.n = n
        self.pool = ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"), initializer=_start)

    def submit(self, out, ks):
        return self.pool.submit(_chunk, out.args, ks)

    def stop(self):
        """Stop what runs now (an error, or frames no longer wanted)."""
        for p in list((getattr(self.pool, "_processes", None) or {}).values()):   # (3.14: terminate_workers)
            p.terminate()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


_OUT = None                 # a worker's Output: the output it plans now


def _start():
    import cv2
    cv2.setNumThreads(2)            # the processes share the CPU; one thread would take other code paths in OpenCV,
                                    # whose results differ by a level here and there from the parent's: two do not


def _chunk(args, ks):
    global _OUT
    if _OUT is None or _OUT.name != args["name"]:
        _OUT = None
        _OUT = Output(**args)
    return [(k, *_OUT.plan(k)) for k in ks], _OUT.notes


def chunks(out, size, ways=1, most=None):
    """The clip's frame indices of `out` cut into runs of `size` frames or a little more, each starting on a frame a
    process can start on (Output.safe); a group of consecutive heavy frames (Output.heavy) is cut into `ways` runs of
    its own (so many workers share it out, each reading its clips on from one frame to the next where a run a frame
    would seek them afresh), `most` frames a run at most, each alone in it where the frame after it can start one."""
    n, a, runs, per = len(out.frames), 0, [], 1
    while a < n:
        b = a + 1
        if not out.heavy(a):
            while b < n and b - a < size and not out.heavy(b):
                b += 1
        else:
            if a == 0 or not out.heavy(a - 1):                   # a group starts: the length of its runs
                g = b
                while g < n and out.heavy(g):
                    g += 1
                per = -(-(g - a) // max(1, ways))
                per = per if most is None else max(1, min(per, most))
            while b < n and b - a < per and out.heavy(b):
                b += 1
        while b < n and not out.safe(b):
            b += 1
        runs.append(range(a, b))
        a = b
    return runs


def _planned(out, jobs, chunk, pool=None):
    """(k, route, rgb24 bytes or mkr-post job, stats) of every frame of `out` in order (Output.plan), from this process
    or from `jobs` workers (`pool`, Planners, else a pool of its own) over runs of about `chunk` frames (chunks): each
    run in its turn, AHEAD runs a worker ahead of the one handed on, and the heavy ones (Output.heavy, a group of them
    in one run a worker) long before theirs, from the start, so their pictures are there when it comes: one worker
    fewer than `jobs` on them at most, and EARLY_BYTES of their pictures not yet handed on. What the runs' cut effects
    settle joins `out`'s notes. Following a render, a run in its turn is planned once the render has its files
    (Output.wait), a heavy one before its turn only when they are there already (Output.ready)."""
    n = len(out.frames)
    if jobs <= 1 or n <= chunk:
        for k in range(n):
            out.wait(k)
            yield (k, *out.plan(k))
        return
    room = EARLY_BYTES // (out.size[0] * out.size[1] * 12)        # heavy frames planned, not yet handed on
    early = max(1, jobs - 1)                                      # workers on heavy runs before their turn
    runs = chunks(out, chunk, early, max(1, room // early))
    heavy = deque(i for i, ks in enumerate(runs) if out.heavy(ks[0]))
    heavy_runs = set(heavy)
    own = pool is None
    if own:
        pool = Planners(jobs)
    work = {}                                                     # run index -> its worker's future
    turn = 0                                                      # the next run planned in its turn
    seen = {note.get("transition") for note in out.notes}

    def submit(at):
        """The runs up to jobs * AHEAD past run `at` in their turn, then heavy runs before theirs: those at work."""
        nonlocal turn
        while turn < len(runs) and turn < at + jobs * AHEAD:
            if turn not in work:
                out.wait(runs[turn][-1])
                work[turn] = pool.submit(out, runs[turn])
            turn += 1
        while heavy and heavy[0] < turn:                          # reached in their turn
            heavy.popleft()
        early = [i for i in work if i >= turn]
        busy = [work[i] for i in early if not work[i].done()]
        held = sum(len(runs[i]) for i in work if i in heavy_runs)
        while (heavy and len(busy) < max(1, jobs - 1) and held + len(runs[heavy[0]]) <= room
               and out.ready(runs[heavy[0]][-1])):
            i = heavy.popleft()
            work[i] = pool.submit(out, runs[i])
            busy.append(work[i])
            held += len(runs[i])
        return busy

    finished_all = False
    try:
        for at in range(len(runs)):
            busy = submit(at)
            while not work[at].done():                            # heavy runs topped up as they finish
                wait([work[at], *busy], return_when=FIRST_COMPLETED)
                busy = submit(at)
            done, notes = work.pop(at).result()
            for note in notes:                                    # what a worker settled that this process did not
                if note.get("transition") not in seen:
                    seen.add(note.get("transition"))
                    out.fx.notes.append(note)
            done.reverse()
            while done:                                           # each frame let go as it is handed on
                yield done.pop()
        finished_all = True
    finally:
        if not finished_all:                                      # an error, or the frames are no longer wanted: stop
            pool.stop()
        elif own:
            pool.close()


class _Sender:
    """mkr-post's jobs sent in order on a thread of their own (finished): a footage op's clip frame decoded and fitted
    first, from clips of its own (mkmmd.postgpu.hand: one decoder a clip, read in order) or taken from `kept` (what an
    output before this one fitted for it), a frame finished here written for mkr-post to pass into the encoder;
    meanwhile the caller takes the next frames from the workers and mkr-post's answers. When it fails it ends
    mkr-post's jobs (mkr-post stops after the frames it has), and `check` raises why."""

    def __init__(self, out, server, kept=None):
        self.out, self.server, self.kept = out, server, kept
        self.clips = FT.source(out.args["proj"].root)
        self.slots = queue.SimpleQueue()
        self.error = None
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="mkpost-send", daemon=True)
        self.thread.start()

    def put(self, slot):
        """Send a planned frame (k, route, job or rgb24 bytes, stats) after the ones put before it."""
        self.check()
        self.slots.put(slot)

    def check(self):
        if self.error is not None:
            raise self.error

    def _run(self):
        try:
            while not self.stopping:
                slot = self.slots.get()
                if slot is None:
                    return
                k, route, payload, stats = slot
                if route == "gpu":
                    if payload["ops"]:
                        PG.hand(self.out.ffx, payload["ops"], self.out.gpu, self.clips, self.kept)
                    self.server.send(payload)
                else:                                             # finished here: through mkr-post, in order
                    path = self.server.folder / f"{os.getpid()}-{k}.done"
                    with open(path, "wb") as fh:
                        fh.write(payload)
                    self.server.send({"k": k, "finished": {"path": str(path), "darkest": stats[0]}})
        except BaseException as e:                                # noqa: BLE001 (raised by check)
            self.error = e
            self.server.end()

    def close(self):
        """Stop after the frame it sends now; mkr-post killed when that takes over SEND_WAIT s (it no longer reads)."""
        self.stopping = True
        self.slots.put(None)
        self.thread.join(SEND_WAIT)
        if self.thread.is_alive():
            self.server.proc.kill()
            self.thread.join(SEND_WAIT)


def finished(out, jobs=1, chunk=None, server=None, pool=None, kept=None):
    """Every frame of `out` (an Output) finished, in order: (k, rgb24 bytes, (darkest luma, screen type frames, kept,
    effects that found a layer missing)), by this process or `jobs` worker processes (_planned; `pool` a Planners the
    run's outputs share) and, with `server` (a mkmmd.postgpu.Server; `out` built with `gpu`), mkr-post for the frames it
    can finish, sent in order by a thread of their own (_Sender; `kept` the clip frames the run's outputs share,
    mkmmd.gpufx.footage.Kept), LOOKAHEAD of them in flight. A server with a sink writes every frame into the encoder
    itself, the ones finished here handed through it in order, SINK_LOOKAHEAD in flight; their bytes come out None."""
    planned = _planned(out, jobs, chunk or CHUNK, pool)
    sink = server is not None and server.sink is not None
    ahead = SINK_LOOKAHEAD if sink else LOOKAHEAD
    sender = None if server is None else _Sender(out, server, kept)
    slots = deque()

    def handed(slot):
        k, route, payload, stats = slot
        if route == "cpu" and not sink:
            return k, payload, stats
        try:
            got_k, dark, data = server.receive()
        except RuntimeError:
            sender.check()                                        # mkr-post stopped after what the sender failed on
            raise
        if got_k != k:
            raise RuntimeError(f"mkr-post answered frame {got_k} for frame {k}")
        return k, data, (stats[0] if route == "cpu" else dark, *stats[1:])

    try:
        for slot in planned:
            k, route, _, stats = slot
            if route == "gpu" or sink:
                sender.put(slot)
                if route == "cpu":
                    slot = (k, route, None, stats)
            slots.append(slot)
            while len(slots) > ahead or (slots and slots[0][1] == "cpu" and not sink):
                yield handed(slots.popleft())
        while slots:
            yield handed(slots.popleft())
    finally:
        planned.close()
        if sender is not None:
            sender.close()
