"""mk doctor: check that everything mk needs is installed and reachable."""
import importlib
import shutil
import subprocess
from pathlib import Path

from .. import bridge
from .. import config as CFG
from .common import CHECK_FAILED, emit

MIN_FREE_GB = 10.0


def add(sub):
    p = sub.add_parser("doctor", help="check the environment (Blender, mmd_tools, tools, disk, keys)",
                       description="Checks Blender + mmd_tools, ffmpeg, espeak-ng, Python packages, free disk space "
                                   "and API keys. Exits 1 if anything required is missing.")
    p.set_defaults(func=run)


def _item(name, ok, detail, required=True):
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def run(args):
    cfg = CFG.load()
    items = []
    bl = Path(cfg["blender"])
    if bl.exists():
        try:
            info = bridge.run("ping", timeout=120)
            items.append(_item("blender", True, f"{info['blender']} (Python {info['python']}, numpy {info['numpy']})"))
            items.append(_item("mmd_tools", info["mmd_tools"], cfg["mmd_addon"]))
        except bridge.BlenderError as e:
            items.append(_item("blender", False, str(e)))
    else:
        items.append(_item("blender", False, f"not found at {bl} (set MK_BLENDER)"))
    for tool, required in (("ffmpeg", True), ("ffprobe", True), ("espeak-ng", False)):
        path = shutil.which(tool)
        items.append(_item(tool, path, path or "not on PATH", required))
    for mod, required in (("numpy", True), ("scipy", True), ("cv2", True), ("PIL", True), ("numba", False)):
        try:
            m = importlib.import_module(mod)
            items.append(_item(f"python:{mod}", True, getattr(m, "__version__", "ok"), required))
        except ImportError:
            items.append(_item(f"python:{mod}", False, "not installed", required))
    assets = Path(cfg["assets"])
    items.append(_item("assets folder", assets.exists(), str(assets), required=False))
    probe = assets if assets.exists() else Path.home()
    free = shutil.disk_usage(probe).free / 1e9
    items.append(_item("disk", free >= MIN_FREE_GB, f"{free:.1f} GB free at {probe} (want {MIN_FREE_GB:.0f}+)",
                       required=False))
    items.append(_item("pexels key", _has_secret("pexels"), "keyring: service pexels key api", required=False))
    ok = all(i["ok"] for i in items if i["required"])
    emit({"ok": ok, "config": cfg, "items": items})
    return 0 if ok else CHECK_FAILED


def _has_secret(service):
    if not shutil.which("secret-tool"):
        return False
    try:
        out = subprocess.run(["secret-tool", "lookup", "service", service, "key", "api"], capture_output=True,
                             timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())
