"""mk gpu: the GPU lock (one GPU job at a time) and what is on the card."""
import argparse
import glob
import os
import re

from .. import gpulock as GL
from .common import UsageError, emit

HELP = """One GPU job at a time (docs/design.md: The GPU lock). The card's memory is shared by every job on it and what does
not fit spills into system memory, which cannot be swapped; so mk render, mk post, mk look, Blender's render jobs, mkr
and mkr-post each hold one lock while they draw, and the others wait their turn.

  mk gpu run [--what TEXT] -- CMD [ARG...]   run CMD holding the lock (waiting for it), for GPU work mk does not start
                                             itself: a Blender script, a scratch mkr build, GPU tests (cargo test)
  mk gpu status                              who holds the lock, the card's memory in use and spilled into system
                                             memory, and the processes using it (JSON)

A command run this way, and everything it starts, holds the lock until the last of them exits
(MK_GPU_LOCK_HELD=1 tells mk's own GPU commands inside it not to wait for it again)."""


def add(sub):
    p = sub.add_parser("gpu", help="the GPU lock: run a command holding it; who holds it, the card's memory",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="gpu_cmd", required=True)
    r = s.add_parser("run", help="run a command holding the GPU lock")
    r.add_argument("--what", help="what the job is, for those who wait (default: the command)")
    r.add_argument("command", nargs=argparse.REMAINDER, help="-- CMD [ARG...]")
    r.set_defaults(func=run)
    st = s.add_parser("status", help="who holds the lock; the card's memory and who uses it")
    st.set_defaults(func=status)


def run(args):
    argv = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not argv:
        raise UsageError("mk gpu run -- CMD [ARG...]: no command given")
    try:
        GL.exec_holding(argv, args.what)
    except FileNotFoundError:
        raise UsageError(f"{argv[0]}: command not found") from None


def _number(path):
    """The integer a sysfs file holds, or None."""
    try:
        with open(path, encoding="ascii") as fh:
            return int(fh.read())
    except (OSError, ValueError):
        return None


def _mb(path):
    n = _number(path)
    return None if n is None else round(n / 2 ** 20)


def _clients():
    """Processes with GPU memory (DRM fdinfo): [{pid, vram_mb, gtt_mb, cmd}], largest first."""
    out = []
    for proc in glob.glob("/proc/[0-9]*"):
        mem = {}
        for f in glob.glob(proc + "/fdinfo/*"):
            try:
                with open(f, encoding="ascii", errors="replace") as fh:
                    txt = fh.read()
            except OSError:
                continue
            cid = re.search(r"drm-client-id:\s*(\d+)", txt)
            if cid is None:
                continue
            vram = re.search(r"drm-memory-vram:\s*(\d+)\s*KiB", txt)
            gtt = re.search(r"drm-memory-gtt:\s*(\d+)\s*KiB", txt)
            mem[cid.group(1)] = (int(vram.group(1)) if vram else 0, int(gtt.group(1)) if gtt else 0)
        if not mem:
            continue
        vram, gtt = sum(v for v, _ in mem.values()), sum(g for _, g in mem.values())
        if vram + gtt < 1024:
            continue
        try:
            with open(proc + "/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()[:200]
        except OSError:
            continue
        out.append({"pid": int(os.path.basename(proc)), "vram_mb": round(vram / 1024), "gtt_mb": round(gtt / 1024),
                    "cmd": cmd})
    return sorted(out, key=lambda c: -(c["vram_mb"] + c["gtt_mb"]))


def status(args):
    cards = []
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if not os.path.exists(dev + "/mem_info_vram_total"):
            continue
        cards.append({"card": dev.split("/")[-2], "vram_used_mb": _mb(dev + "/mem_info_vram_used"),
                      "vram_total_mb": _mb(dev + "/mem_info_vram_total"),
                      "spilled_mb": _mb(dev + "/mem_info_gtt_used"),
                      "busy_percent": _number(dev + "/gpu_busy_percent")})
    emit({"lock": str(GL.path()), "held_by": GL.holder(), "cards": cards, "processes": _clients()})
    return 0
