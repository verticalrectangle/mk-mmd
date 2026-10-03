"""Job runner inside Blender: reads job.json, runs the registered op, writes result.json. Also the `mk serve` loop."""
import importlib
import json
import os
import pkgutil
import socket
import sys
import time
import traceback

from ..core import jsonx

OPS = {}
CTX = {"project": None, "config": {}, "mmd_tools": False}


def op(name):
    """Register a Blender-side handler: fn(args: dict) -> JSON-able data."""
    def deco(fn):
        OPS[name] = fn
        return fn
    return deco


def _load_ops():
    import mkmmd.blender as pkg
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("ops_"):
            importlib.import_module(f"mkmmd.blender.{mod.name}")


def enable_mmd(addon):
    import addon_utils
    try:
        addon_utils.enable(addon, default_set=True)
        return True
    except Exception:  # noqa: BLE001 - reported by `mk doctor`
        return False


def run_job(job):
    t0 = time.time()
    CTX["project"] = job.get("project")
    CTX["config"] = job.get("config") or {}
    try:
        name = job["op"]
        if name not in OPS:
            raise KeyError(f"unknown op {name!r} (known: {', '.join(sorted(OPS))})")
        data = OPS[name](job.get("args") or {})
        return {"ok": True, "data": jsonx.to_jsonable(data), "seconds": round(time.time() - t0, 3)}
    except Exception as e:  # noqa: BLE001 - the CLI shows the error and trace
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(),
                "seconds": round(time.time() - t0, 3)}


def _write(path, res):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    job_path, result_path = argv[0], argv[1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    cfg = job.get("config") or {}
    CTX["mmd_tools"] = enable_mmd(cfg.get("mmd_addon", "bl_ext.user_default.mmd_tools"))
    _load_ops()
    if job["op"] == "__serve__":
        serve(job["args"]["socket"], result_path)
        return
    _write(result_path, run_job(job))


def serve(sock_path, ready_path):
    """Answer newline-delimited JSON jobs on a Unix socket until a {"op": "__stop__"} job arrives."""
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(4)
    _write(ready_path, {"ok": True, "data": {"socket": sock_path, "pid": os.getpid()}})
    print(f"MK_SERVE ready {sock_path}", flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(1 << 20)
                    if not chunk:
                        break
                    buf += chunk
                if not buf:
                    continue
                job = json.loads(buf.decode("utf-8"))
                if job.get("op") == "__stop__":
                    conn.sendall(b'{"ok": true, "data": "stopped"}\n')
                    break
                res = run_job(job)
                conn.sendall(json.dumps(res, ensure_ascii=False).encode("utf-8") + b"\n")
    finally:
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
