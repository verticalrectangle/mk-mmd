"""One GPU job at a time (docs/design.md: The GPU lock). The card's memory is shared by every job on it, and what does not
fit spills into system memory, which cannot be swapped: two renders at once can take the machine down. So every job
that draws on the GPU (mk render, mk post, mk look, Blender's render jobs, mkr, mkr-post) holds one lock while it runs
and the others wait their turn.

The lock is an advisory `flock` on `$MK_GPU_LOCK`, else `<cache>/gpu.lock` (config.cache_dir: `$MK_CACHE`, default
~/.cache/mk); the kernel releases it when its holder exits, however it exits. The holder writes its pid and what it is
into the file, for those that wait. Jobs are served first come first served: one that finds the GPU busy (or others
already waiting) puts a ticket `<ns>-<pid>` into the queue folder beside the lock file (`gpu.lock.queue/`) and only the
oldest live ticket waits on the lock itself; a dead job's ticket is dropped by whoever sees it. A job's children
(subprocesses, worker processes, mkr) run inside its hold: it sets `MK_GPU_LOCK_HELD=1` while it holds the lock, and
anything that finds it set skips the lock (mkr and mkr-post do the same in Rust: crate mkr-gpulock)."""
import contextlib
import fcntl
import os
import sys
import time
from pathlib import Path

from . import config as CFG
from . import memory as MEM

ENV_HELD = "MK_GPU_LOCK_HELD"
ENV_PATH = "MK_GPU_LOCK"
POLL_S = 0.2                                  # how often a job in the queue looks whether it is first


def path():
    """The lock file."""
    given = os.environ.get(ENV_PATH)
    return Path(given).expanduser() if given else CFG.cache_dir() / "gpu.lock"


def holder():
    """Who holds the GPU, as its holder wrote it ("<pid> <what>"), or None when nobody does."""
    p = path()
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return os.pread(fd, 4096, 0).decode("utf-8", "replace").strip() or "a process that did not say"
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def _line(queue):
    """The tickets of the jobs waiting, oldest first [(ns, pid, name)]; a dead job's ticket is removed."""
    out = []
    for name in os.listdir(queue):
        try:
            ns, pid = (int(x) for x in name.split("-", 1))
        except ValueError:
            continue
        if not MEM.alive(pid):
            with contextlib.suppress(OSError):
                os.unlink(queue / name)
            continue
        out.append((ns, pid, name))
    return sorted(out)


def _acquire(what):
    """The lock file's descriptor, holding the lock (waited for in turn, said once on stderr with who has it and the
    place in the queue), its holder written in."""
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    queue = Path(f"{p}.queue")
    queue.mkdir(exist_ok=True)
    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        got = False
        if not _line(queue):                  # nobody waiting: take it if it is free
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
            except BlockingIOError:
                pass
        if not got:
            ticket = queue / f"{time.time_ns():020d}-{os.getpid()}"
            ticket.touch(exist_ok=False)
            try:
                who = os.pread(fd, 4096, 0).decode("utf-8", "replace").strip() or "another job"
                place = next((i + 1 for i, (_, _, n) in enumerate(_line(queue)) if n == ticket.name), 1)
                print(f"mk: waiting for the GPU, held by {who} ({place} in line)", file=sys.stderr, flush=True)
                t0 = time.time()
                while True:
                    if not ticket.exists():           # someone cleared the queue: back in at the same place
                        ticket.touch()
                    line = _line(queue)
                    if line and line[0][2] == ticket.name:
                        fcntl.flock(fd, fcntl.LOCK_EX)    # first in line: the next turn is this job's
                        break
                    time.sleep(POLL_S)
            finally:
                with contextlib.suppress(OSError):
                    ticket.unlink()
            print(f"mk: got the GPU after {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()} {what}\n".encode(), 0)
    except BaseException:
        os.close(fd)
        raise
    return fd


@contextlib.contextmanager
def held(what):
    """Hold the GPU for the `with` block, waiting until it is free; inside a process tree that already holds it,
    nothing."""
    if os.environ.get(ENV_HELD) == "1":
        yield
        return
    fd = _acquire(what)
    os.environ[ENV_HELD] = "1"
    try:
        yield
    finally:
        os.environ.pop(ENV_HELD, None)
        os.ftruncate(fd, 0)
        os.close(fd)                                   # closing the file releases the lock


def exec_holding(argv, what=None):
    """Become `argv` (os.execvp) holding the GPU: the lock's descriptor is inherited by the command and everything it
    starts, so the lock lasts until the last of them exits (`mk gpu run`)."""
    if os.environ.get(ENV_HELD) != "1":
        fd = _acquire(what or " ".join(argv))
        os.set_inheritable(fd, True)
        os.environ[ENV_HELD] = "1"
    os.execvp(argv[0], argv)
