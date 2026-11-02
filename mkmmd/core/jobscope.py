"""Blender jobs in a systemd scope of their own (docs/design.md: The bridge). Stdlib only: the CLI's bridge imports it,
and so does a build that starts another Blender (the breathe probe).

A Blender that mk starts otherwise runs in the cgroup of whatever started mk, often a terminal app's systemd service.
When memory runs out, the kernel kills the biggest process, a Blender job, and the service's default `OOMPolicy=stop` then
stops that whole service: the terminal and everything in it close with the job. Run as a transient scope of its own
(`systemd-run --user --scope`), the job is its own unit: an overflow kills that job alone, and its memory is capped
(`MemoryMax`, a share of the machine's RAM), so one job cannot take the machine down by itself."""
import os
import shutil
import subprocess

SHARE = 0.6                  # the share of the RAM a job may hold, unless the config says otherwise
_usable = None               # whether systemd-run --user --scope runs a command here (probed once per process)


class JobScopeError(ValueError):
    pass


def total():
    """Bytes of RAM (MemTotal), or None where the system does not say."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            return next(int(line.split()[1]) * 1024 for line in fh if line.startswith("MemTotal:"))
    except (OSError, StopIteration, ValueError):
        return None


def share(value=None):
    """The share of the RAM a job may hold, from the config's `blender_memory` (or MK_BLENDER_MEMORY): a number in
    (0, 1], or 0 / "off" for no scope at all; None is SHARE."""
    if value is None:
        return SHARE
    if isinstance(value, str) and value.strip().lower() in ("off", "no", "false"):
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise JobScopeError(f"blender_memory is a share of the RAM in (0, 1], or 0 / \"off\": not {value!r}") from None
    if not 0.0 <= v <= 1.0:
        raise JobScopeError(f"blender_memory is a share of the RAM in (0, 1], or 0 / \"off\": not {value!r}")
    return v


def usable():
    """Whether `systemd-run --user --scope` can run a command here (Linux, systemd-run, a user manager to talk to)."""
    global _usable
    if _usable is None:
        exe = shutil.which("systemd-run")
        _usable = False
        if exe and os.environ.get("XDG_RUNTIME_DIR"):
            try:
                r = subprocess.run([exe, "--user", "--scope", "--quiet", "--collect", "true"], capture_output=True,
                                   timeout=30)
                _usable = r.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                _usable = False
    return _usable


def wrap(cmd, what, mem_share=SHARE):
    """`cmd` (a list) run in a transient scope of its own (`what` describes it to systemd) with its memory capped at
    `mem_share` of the RAM; `cmd` as it is when `mem_share` is 0 or no scope can be made here. The scope runs the
    command in the foreground: its output, exit status and children are the command's own."""
    if not mem_share or not usable():
        return list(cmd)
    out = [shutil.which("systemd-run"), "--user", "--scope", "--quiet", "--collect", f"--description=mk {what}"]
    ram = total()
    if ram:
        out += ["-p", f"MemoryMax={int(ram * mem_share)}"]
    return out + ["--", *[str(c) for c in cmd]]


def cap_gb(mem_share=SHARE):
    """The cap in GB (one decimal) a job runs under, or None (no scope, or the RAM unknown)."""
    ram = total()
    return round(ram * mem_share / 1e9, 1) if ram and mem_share and usable() else None


def killed_message(returncode, what, mem_share=SHARE):
    """What to tell the user when a job's Blender ended without a result, from its exit status: a kill (SIGKILL) is most
    likely the memory running out; None for any other end."""
    if returncode not in (-9, 137):
        return None
    cap = cap_gb(mem_share)
    held = f" (a job may hold {cap} GB: `blender_memory` = {mem_share} of the RAM)" if cap else ""
    return (f"Blender {what} was killed: most likely the memory ran out{held}. Close other heavy jobs (other builds, "
            f"renders) or raise `blender_memory` in ~/.config/mk/config.toml, and run it again")
