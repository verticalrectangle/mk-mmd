"""Memory for worker pools and scratch files: what the system has to spare, and scratch in memory-backed folders
(/tmp and /dev/shm are RAM here) that outlives a run killed outright, which never reaches its own cleanup."""
import os
import shutil


def available():
    """Bytes of memory available without swapping (MemAvailable), or None where the system does not say."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            return next(int(line.split()[1]) * 1024 for line in fh if line.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError):
        return None


def scratch_prefix(name):
    """The prefix of a scratch file or folder `name` makes: `<name>_<pid>_`, so `sweep` can tell whose it is."""
    return f"{name}_{os.getpid()}_"


def alive(pid):
    """Whether process `pid` still runs (a zombie, exited but not yet reaped by its parent, does not)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return fh.read().rsplit(b")", 1)[1].split()[0] != b"Z"
    except FileNotFoundError:
        return False
    except (OSError, IndexError):             # no /proc: ask the kernel
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:               # someone else's process: alive
            return True
        return True


def sweep(name, folders):
    """Remove the scratch `name` left in `folders` by processes that are gone (see `scratch_prefix`); names without a
    process id are left alone. Returns how many were removed."""
    head, n = f"{name}_", 0
    for folder in dict.fromkeys(folders):
        try:
            entries = os.listdir(folder)
        except OSError:
            continue
        for entry in entries:
            pid = entry[len(head):].split("_", 1)[0] if entry.startswith(head) else ""
            if not pid.isdigit() or alive(int(pid)):
                continue
            path = os.path.join(folder, entry)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.unlink(path)
                except OSError:
                    continue
            n += 1
    return n
