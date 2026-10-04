"""mk doctor: check that everything mk needs is installed and reachable."""
import importlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from .. import bridge
from .. import config as CFG
from .common import CHECK_FAILED, emit

MIN_FREE_GB = 10.0


def add(sub):
    p = sub.add_parser("doctor", help="check the environment (Blender, mmd_tools, tools, disk, keys)",
                       description="Checks Blender + mmd_tools (and the Python wheels the extension bundles), ffmpeg, "
                                   "ffprobe, espeak-ng, Python packages, free disk space and API keys. Exits 1 if "
                                   "anything required is missing: Blender, mmd_tools, ffmpeg, ffprobe, numpy, scipy, "
                                   "opencv, pillow. Optional (reported, never fail the exit code): espeak-ng (lip sync "
                                   "from sung words), numba (faster solvers), mediapipe (mk ref), torch, demucs and "
                                   "faster-whisper (mk timeline), the assets folder, 10 GB free disk, the Pexels key "
                                   "(mk ref search). --fix reinstalls missing mmd_tools wheels (a Blender started "
                                   "with --factory-startup deletes them, and PMX import then fails).")
    p.add_argument("--fix", action="store_true", help="reinstall mmd_tools' bundled wheels that are missing")
    p.set_defaults(func=run)


def extension_wheels(version):
    """[(wheel path, top-level package names, installed?, site-packages dir)] for the wheels mmd_tools bundles,
    checked against the extensions' site-packages of Blender `version` ("4.2")."""
    base = Path.home() / ".config" / "blender" / version / "extensions"
    site = base / ".local" / "lib"
    sites = sorted(site.glob("python3.*/site-packages"))
    out = []
    for whl in sorted(base.glob("*/mmd_tools/wheels/*.whl")):
        with zipfile.ZipFile(whl) as z:
            tops = [n for n in z.namelist() if n.endswith(".dist-info/top_level.txt")]
            names = z.read(tops[0]).decode().split() if tops else [whl.name.split("-")[0]]
        ok = bool(sites) and all(any((s / n).exists() or (s / f"{n}.py").exists() for s in sites) for n in names)
        out.append((whl, names, ok, sites[0] if sites else None))
    return out


def _fix_wheels(version):
    fixed = []
    for whl, names, ok, site in extension_wheels(version):
        if ok or site is None:
            continue
        with zipfile.ZipFile(whl) as z:
            z.extractall(site)
        fixed.append(whl.name)
    return fixed


def _item(name, ok, detail, required=True):
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def run(args):
    cfg = CFG.load()
    items = []
    bl = Path(cfg["blender"])
    if bl.exists():
        m = re.search(r"blender-(\d+\.\d+)", str(bl))
        version = m.group(1) if m else "4.2"
        if getattr(args, "fix", False):
            fixed = _fix_wheels(version)
            if fixed:
                items.append(_item("mmd_tools wheels reinstalled", True, ", ".join(fixed), required=False))
        for whl, names, ok, site in extension_wheels(version):
            items.append(_item(f"mmd_tools wheel:{'/'.join(names)}", ok,
                               str(site) if ok else f"missing from {site} (run mk doctor --fix)"))
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
    for mod, required in (("numpy", True), ("scipy", True), ("cv2", True), ("PIL", True), ("numba", False),
                          ("mediapipe", False), ("torch", False), ("demucs", False), ("faster_whisper", False)):
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
