"""Run Blender jobs from the CLI (see docs/design.md: The bridge)."""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import config as CFG

ROOT = Path(__file__).resolve().parent.parent          # the folder that contains the mkmmd package
# ops that leave the scene as they found it: safe on `mk serve`
READONLY_OPS = {"ping", "list", "q", "sample", "visibility", "hand_model"}

BOOT = ("import sys; sys.path.insert(0, {root!r}); "
        "from mkmmd.blender import runtime; runtime.main()")


class BlenderError(RuntimeError):
    def __init__(self, message, log=None, trace=None):
        super().__init__(message)
        self.log = log
        self.trace = trace


def _tail(path, n=40):
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def serve_socket(blend):
    """Unix socket path of the `mk serve` process for this scene."""
    h = hashlib.sha1(str(Path(blend).resolve()).encode()).hexdigest()[:16]
    d = CFG.cache_dir() / "serve"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.sock"


def _send(sock_path, job, timeout):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(sock_path))
        s.sendall(json.dumps(job, ensure_ascii=False).encode("utf-8") + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode("utf-8"))


def empty_blend(cfg):
    """A factory-default empty scene that jobs without a .blend start from. Never start job Blenders with
    --factory-startup: without the user's preferences Blender sees mmd_tools as disabled and its extension sync
    removes the extension's wheels (opencc), which breaks PMX import in every Blender running at the time. This file
    is made once, by a Blender whose extension and config folders point at a scratch place."""
    path = CFG.cache_dir() / "empty.blend"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"empty.{os.getpid()}.blend")
    scratch = CFG.cache_dir() / "factory"
    env = dict(os.environ, BLENDER_USER_EXTENSIONS=str(scratch / "extensions"), BLENDER_USER_CONFIG=str(scratch / "config"),
               BLENDER_USER_SCRIPTS=str(scratch / "scripts"))
    expr = ("import bpy; bpy.ops.wm.read_homefile(use_factory_startup=True, use_empty=True); "
            f"bpy.ops.wm.save_as_mainfile(filepath={str(tmp)!r}, compress=False)")
    r = subprocess.run([cfg["blender"], "-b", "--factory-startup", "--python-expr", expr], env=env,
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not tmp.exists():
        raise BlenderError(f"could not create {path}: {(r.stdout + r.stderr)[-1500:]}")
    os.replace(tmp, path)
    return path


def blender_command(job_path, result_path, blend=None, cfg=None):
    cfg = cfg or CFG.load()
    cmd = [cfg["blender"], "-b", str(blend or empty_blend(cfg))]
    cmd += ["-y", "--python-exit-code", "3", "--python-expr", BOOT.format(root=str(ROOT)), "--",
            str(job_path), str(result_path)]
    return cmd


def run(op, args=None, blend=None, project=None, timeout=None, use_serve=True):
    """Run one Blender op and return its data. Raises BlenderError with the log tail on failure."""
    cfg = CFG.load()
    if not Path(cfg["blender"]).exists():
        raise BlenderError(f"Blender not found at {cfg['blender']} (set MK_BLENDER or ~/.config/mk/config.toml)")
    if blend is not None:
        blend = Path(blend).expanduser().resolve()
        if not blend.exists():
            raise BlenderError(f"{blend} does not exist")
    job = {"op": op, "args": args or {}, "project": project.to_job() if project else None,
           "config": {"mmd_addon": cfg["mmd_addon"], "assets": cfg["assets"], "python": sys.executable}}
    if use_serve and blend is not None and op in READONLY_OPS:
        sock = serve_socket(blend)
        if sock.exists():
            try:
                res = _send(sock, job, timeout or 3600)
                return _unwrap(res, None)
            except (ConnectionRefusedError, FileNotFoundError, socket.timeout):
                pass
    jobdir = CFG.cache_dir() / "jobs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{op}-{uuid.uuid4().hex[:6]}"
    jobdir.mkdir(parents=True)
    job_path, result_path, log_path = jobdir / "job.json", jobdir / "result.json", jobdir / "blender.log"
    job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    cmd = blender_command(job_path, result_path, blend, cfg)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            raise BlenderError(f"Blender job {op!r} timed out after {timeout} s", log=str(log_path))
    if not result_path.exists():
        raise BlenderError(f"Blender exited (code {proc.returncode}) without a result for {op!r}\n"
                           f"{_tail(log_path)}", log=str(log_path))
    res = json.loads(result_path.read_text(encoding="utf-8"))
    data = _unwrap(res, log_path)
    if not os.environ.get("MK_KEEP_JOBS"):
        shutil.rmtree(jobdir, ignore_errors=True)
    return data


def _unwrap(res, log_path):
    if not res.get("ok"):
        raise BlenderError(res.get("error", "unknown error"), log=str(log_path) if log_path else None,
                           trace=res.get("trace"))
    return res.get("data")
