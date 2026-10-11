"""mk's own renderer, the run (docs/design.md: Rendering): a server of its own on 127.0.0.1 (mkmmd.site.server, a page
of its own: web/render.html) hands headless Chromium the bake and the frames of a plan (mkmmd.render.plan); the page
draws each frame's passes on the GPU and posts them back, and worker processes finish them (mkmmd.render.compose)
while it draws the next. The page waits while the workers are full, so a frame's passes are never held for long."""
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..site import server as SV
from . import compose as CO

BROWSERS = ("chromium", "chromium-browser", "google-chrome-stable", "google-chrome")
FLAGS = ["--headless=new", "--use-angle=gl", "--enable-gpu", "--ignore-gpu-blocklist", "--no-first-run",
         "--no-default-browser-check", "--disable-extensions", "--mute-audio", "--disable-background-timer-throttling"]
MAX_BODY = 1 << 30
FIRST_S, STALL_S = 300.0, 60.0          # seconds the page may take for its first frame (loading the bake), then each


class EngineError(RuntimeError):
    """The renderer could not draw its frames."""


def browser():
    """The Chromium to draw with: $MK_CHROMIUM, else the first of BROWSERS on the PATH; None without one."""
    env = os.environ.get("MK_CHROMIUM")
    if env:
        return env
    return next((p for p in (shutil.which(b) for b in BROWSERS) if p), None)


def run(bake_dir, plan, paths, workers=None):
    """Draw the plan's items, item i into paths[i]; the frames on disk already must be left out of the plan. Returns
    {frames, seconds, gpu_seconds, renderer, workers}. A page that sends no frame for STALL_S seconds (FIRST_S for the
    first, while it loads the bake) has stopped: the run fails, naming the frame it was on."""
    items = plan["items"]
    if not items:
        return {"frames": 0, "seconds": 0.0}
    exe = browser()
    if exe is None:
        raise EngineError("mk's renderer needs Chromium (chromium on the PATH, or $MK_CHROMIUM)")
    bake_dir = Path(bake_dir)
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    room = threading.Semaphore(workers * 2)
    state = {"done": threading.Event(), "result": None, "error": None, "finished": 0, "sent": 0, "last": time.time()}
    lock = threading.Lock()
    looks = plan["looks"]
    page_job = {"files": {k: str(bake_dir / f) for k, f in (("json", "bake.json"), ("bin", "bake.bin"), ("glb", "scene.glb"))},
                "looks": [{k: lk[k] for k in ("draw", "ids", "light", "tex")} for lk in looks],
                "items": [{k: it[k] for k in ("output", "frame", "k", "camera", "look", "size")} for it in items]}

    def fail(msg):
        with lock:
            if state["error"] is None:
                state["error"] = msg
        state["done"].set()

    def finished(fut):
        room.release()
        e = fut.exception()
        if e is not None:
            fail(f"composing a frame failed: {type(e).__name__}: {e}")
            return
        with lock:
            state["finished"] += 1

    def job_route(site, query, body):
        return 200, page_job

    def frame_route(site, query, body):
        if state["error"] is not None:
            raise SV.HttpError(409, state["error"])
        i = int(query.get("i", -1))
        if not 0 <= i < len(items):
            raise SV.HttpError(400, f"no item {i}")
        it = items[i]
        state["sent"], state["last"] = i + 1, time.time()
        lk = looks[it["look"]]
        room.acquire()                                     # the page waits while every worker is busy
        fut = pool.submit(CO.finish, str(paths[i]), body, it["size"], lk["tex"], lk["spec"], lk["table"],
                          plan["rings"], it["t"], it["centres"])
        fut.add_done_callback(finished)
        return 200, {"ok": True}

    def done_route(site, query, body):
        msg = SV.body_json(body) or {}
        if msg.get("error"):
            fail(f"the page stopped: {msg['error']}")
        else:
            state["result"] = msg
            state["done"].set()
        return 200, {"ok": True}

    site = SV.Site(routes={("GET", "render/job"): job_route, ("POST", "render/frame"): frame_route,
                           ("POST", "render/done"): done_route}, max_body=MAX_BODY)
    site.allow(bake_dir)
    httpd = SV.serve(site)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}/{site.token}/render"
    t0 = time.time()
    profile = tempfile.mkdtemp(prefix="mk_render_chromium_")
    errlog = open(os.path.join(profile, "stderr.log"), "wb")
    proc = subprocess.Popen([exe, *FLAGS, f"--user-data-dir={profile}", url], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=errlog)
    try:
        while not state["done"].wait(1.0):
            quiet = time.time() - state["last"]
            if proc.poll() is not None:
                fail(f"Chromium quit ({proc.returncode})")
            elif quiet > (STALL_S if state["sent"] else FIRST_S):
                it = items[min(state["sent"], len(items) - 1)]
                fail(f"the page stopped drawing: nothing for {quiet:.0f} s, at {it['output']} frame {it['frame']}")
        pool.shutdown(wait=True)
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        errlog.close()
        tail = Path(profile, "stderr.log").read_text(encoding="utf-8", errors="replace")[-2000:]
        httpd.shutdown()
        httpd.server_close()
        pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(profile, ignore_errors=True)
    if state["error"] is not None:
        raise EngineError(state["error"] + (f"\n{tail}" if "Chromium quit" in state["error"] else ""))
    res = state["result"] or {}
    return {"frames": state["finished"], "seconds": round(time.time() - t0, 1),
            "gpu_seconds": round(float(res.get("gpu_seconds", 0.0)), 1), "renderer": res.get("renderer"),
            "workers": workers}
