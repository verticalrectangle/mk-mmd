"""mk serve: keep one Blender process alive with a scene loaded, for fast repeated read-only commands."""
import json
import os
import subprocess
import time

from .. import bridge
from .. import config as CFG
from .common import emit, get_project, scene_path, UsageError, add_project_arg


def add(sub):
    p = sub.add_parser(
        "serve", help="keep a scene loaded in a background Blender (fast q, check, grip)",
        description="Starts Blender with SCENE loaded and answers the read-only jobs (ping, list, q, sample, visibility, "
                    "hand_model: what `mk q`, `mk check` and `mk grip` send) on a Unix socket; those commands use it "
                    "automatically, which matters for big scenes. Jobs that change the scene (`mk look` hides objects "
                    "and adds a camera, `mk build`, `mk render`) always get a fresh Blender. Runs in the foreground: "
                    "start it as a background job. `mk serve SCENE --stop` shuts it down. Re-run after the scene file "
                    "changes on disk.")
    p.add_argument("scene", nargs="?", metavar="SCENE.blend", help="default: the project's scene")
    p.add_argument("--stop", action="store_true", help="stop the server for SCENE")
    add_project_arg(p)
    p.set_defaults(func=run)


def run(args):
    proj = get_project(args)
    blend = scene_path(args.scene, proj)
    sock = bridge.serve_socket(blend)
    if args.stop:
        if not sock.exists():
            raise UsageError(f"no server running for {blend}")
        res = bridge._send(sock, {"op": "__stop__"}, 30)
        emit({"stopped": str(blend), "reply": res})
        return 0
    cfg = CFG.load()
    jobdir = CFG.cache_dir() / "jobs" / f"serve-{int(time.time())}"
    jobdir.mkdir(parents=True)
    job_path, ready_path = jobdir / "job.json", jobdir / "ready.json"
    job = {"op": "__serve__", "args": {"socket": str(sock)}, "project": proj.to_job() if proj else None,
           "config": {"mmd_addon": cfg["mmd_addon"], "assets": cfg["assets"]}}
    job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    cmd = bridge.blender_command(job_path, ready_path, blend, cfg)
    with open(jobdir / "blender.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
        for _ in range(600):
            if ready_path.exists() or proc.poll() is not None:
                break
            time.sleep(0.2)
        if not ready_path.exists():
            proc.kill()
            raise bridge.BlenderError(f"serve failed to start for {blend}", log=str(jobdir / "blender.log"))
        print(json.dumps({"serving": str(blend), "socket": str(sock), "pid": proc.pid}), flush=True)
        return proc.wait()
