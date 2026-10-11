"""The site's server (docs/design.md: The site): Python's standard library, on this machine only.

One server per user (per MK_CACHE) serves every page `mk site` and `mk review open` ask for. It listens on 127.0.0.1 at
a free port and every URL starts with a random token (`/<token>/...`), so no other web page can read or write through
it; a request must name the server itself in its Host header (no DNS rebinding). Files come only from the folders pages
were opened for (a review's folder, a project's root), the page's own files (`web/`) and mk's site cache. The registry
(`<cache>/site/server.json`: pid, port, token) lets the next command find the running server; it stops after `idle`
seconds without a request (an open page pings every 30 s).
"""
import email.utils
import fcntl
import json
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import __version__
from .. import config as CFG

WEB = Path(__file__).with_name("web")
IDLE_S = 3 * 3600
MAX_BODY = 64 << 20
CHUNK = 1 << 20
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".glb": "model/gltf-binary", ".png": "image/png", ".jpg": "image/jpeg",
         ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif", ".svg": "image/svg+xml",
         ".flac": "audio/flac", ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".ogg": "audio/ogg",
         ".mp4": "video/mp4", ".bin": "application/octet-stream"}


class HttpError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def cache():
    path = CFG.cache_dir() / "site"
    path.mkdir(parents=True, exist_ok=True)
    return path


class Site:
    """What one server knows: its token, the folders it may serve, the routes of its API, when it was last asked."""

    def __init__(self, token=None, idle=IDLE_S, routes=None):
        self.token = token or secrets.token_urlsafe(18)
        self.idle = idle
        self.routes = routes or {}
        self.roots = set()
        self.lock = threading.Lock()
        self.last = time.monotonic()
        self.state = {}          # what the API keeps between requests (build locks, decoded indexes ...)

    def allow(self, folder):
        with self.lock:
            self.roots.add(Path(folder).expanduser().resolve())

    def allowed(self, path):
        """`path` resolved, when it is a file in an allowed folder, the page's own files or the site cache."""
        if not path:
            raise HttpError(400, "no path")
        p = Path(path).expanduser().resolve()
        with self.lock:
            roots = [*self.roots, WEB.resolve(), cache().resolve()]
        if not any(p == r or r in p.parents for r in roots):
            raise HttpError(403, f"{p} is outside the folders this site serves")
        if not p.is_file():
            raise HttpError(404, f"no file {p}")
        return p


def body_json(body):
    try:
        return json.loads(body or b"null")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HttpError(400, f"not JSON: {e}")


class Handler(BaseHTTPRequestHandler):
    server_version = "mk-site/" + __version__
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("HEAD")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def _dispatch(self, method):
        site = self.server.site
        site.last = time.monotonic()
        try:
            port = self.server.server_port
            if self.headers.get("Host", "") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                raise HttpError(403, "this server answers to 127.0.0.1 only")
            parts = urllib.parse.urlsplit(self.path)
            segs = parts.path.split("/", 2)
            if len(segs) < 2 or not secrets.compare_digest(segs[1].encode(), site.token.encode()):
                raise HttpError(404, "not found")
            rest = urllib.parse.unquote(segs[2]) if len(segs) > 2 else ""
            query = {k: v[-1] for k, v in urllib.parse.parse_qs(parts.query, keep_blank_values=True).items()}
            if method in ("GET", "HEAD"):
                if rest in ("", "index.html"):
                    return self._file(WEB / "index.html", head=method == "HEAD")
                if rest.startswith("app/"):
                    return self._file(self._web(rest[4:]), head=method == "HEAD")
                if rest == "file":
                    return self._file(site.allowed(query.get("path")), head=method == "HEAD")
            if rest.startswith("api/"):
                fn = site.routes.get(("GET" if method == "HEAD" else method, rest[4:]))
                if fn is None:
                    raise HttpError(404, f"no {method} {rest}")
                body = self._body() if method in ("POST", "PUT") else None
                status, payload = fn(site, query, body)
                return self._json(status, payload)
            raise HttpError(404, f"no {rest}")
        except HttpError as e:
            self._json(e.status, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001 - every failure goes back to the page, named
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def _web(self, name):
        p = (WEB / name).resolve()
        if WEB.resolve() not in p.parents or not p.is_file():
            raise HttpError(404, f"no app/{name}")
        return p

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise HttpError(413, f"a body is at most {MAX_BODY >> 20} MB")
        return self.rfile.read(n) if n else b""

    def _json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path, head=False):
        st = path.stat()
        size, modified = st.st_size, email.utils.formatdate(st.st_mtime, usegmt=True)
        if self.headers.get("If-Modified-Since") == modified and not self.headers.get("Range"):
            self.send_response(304)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = 0, size - 1
        rng = self.headers.get("Range") or ""
        if rng.startswith("bytes=") and "," not in rng and size:
            a, _, b = rng[6:].partition("-")
            if a.strip():
                start, end = int(a), min(int(b), size - 1) if b.strip() else size - 1
            elif b.strip():
                start = max(0, size - int(b))
            if start > end:
                raise HttpError(416, f"range {rng} is outside {size} bytes")
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        kind = TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", kind)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1 if size else 0))
        self.send_header("Last-Modified", modified)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if head or not size:
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def serve(site, port=0):
    """A server for `site` on 127.0.0.1 (port 0: any free one); call serve_forever on it."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    httpd.site = site
    return httpd


# ------------------------------------------------------------------------------------------------------------ registry
def registry():
    return cache() / "server.json"


def ping(port, token, timeout=1.0):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/{token}/api/ping", timeout=timeout) as r:
            return json.loads(r.read()).get("ok") is True
    except (OSError, ValueError):
        return False


def running():
    """The running server's base URL (`http://127.0.0.1:PORT/TOKEN/`), or None."""
    try:
        rec = json.loads(registry().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if ping(rec.get("port"), rec.get("token")):
        return f"http://127.0.0.1:{rec['port']}/{rec['token']}/"
    return None


def ensure(timeout=20.0):
    """The running server's base URL, starting one (detached, logging to `<cache>/site/server.log`) when none runs."""
    with open(cache() / "start.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        base = running()
        if base:
            return base
        log = open(cache() / "server.log", "ab")
        proc = subprocess.Popen([sys.executable, "-m", "mkmmd.site.server"], stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True, cwd=str(Path.home()))
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            base = running()
            if base:
                return base
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        tail = (cache() / "server.log").read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"the site server did not start: {tail}")


def stop():
    """Stop the running server; True when one was running."""
    try:
        rec = json.loads(registry().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not ping(rec.get("port"), rec.get("token")):
        return False
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{rec['port']}/{rec['token']}/api/stop", data=b"{}", method="POST")
        urllib.request.urlopen(req, timeout=2.0).read()
    except OSError:
        pass
    return True


def main():
    from . import api
    site = Site(routes=api.routes())
    httpd = serve(site)
    site.state["httpd"] = httpd
    rec = {"pid": os.getpid(), "port": httpd.server_port, "token": site.token, "version": __version__,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    registry().write_text(json.dumps(rec), encoding="utf-8")

    def watch():
        while True:
            time.sleep(30)
            if time.monotonic() - site.last > site.idle:
                httpd.shutdown()
                return

    threading.Thread(target=watch, daemon=True).start()
    try:
        httpd.serve_forever()
    finally:
        try:
            if json.loads(registry().read_text(encoding="utf-8")).get("pid") == os.getpid():
                registry().unlink()
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    main()
