"""Open a site page: beside the agent's pane in Tern (a browser block docked to its right, below it or in a new tab),
in place of a pane (the plugin opening a review file from Tern's Files pane), or in the system's browser outside Tern.

A review's Send reaches the agent: `mk review open --wait` waiting for it returns it (`waiting` tells the server it
waits), else the server types the answers into the Tern pane of the agent that opened the page (`tell`). An agent is
known by its shell's environment (AGENT_VARS): a page a person opens from their own shell has none, and Send only
saves there."""
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from . import server as S

AGENT_VARS = ("AGENT", "CLAUDECODE", "OMPCODE")      # set in the shell commands of a coding agent


def opener():
    """The Tern pane of the agent running this command ({pane, window}), or None: outside Tern, or a person's shell."""
    pane = os.environ.get("TERN_PANE", "")
    if not (pane.isdigit() and shutil.which("tern") and any(os.environ.get(v) for v in AGENT_VARS)):
        return None
    return {"pane": pane, "window": os.environ.get("TERN_WINDOW_KEY", "")}


def page(**what):
    """The URL of a page for `review=PATH` or `project=DIR`, from the running server (started when none runs). The
    server learns which agent opened it (`opener`), to tell it when a review is sent."""
    base = S.ensure()
    req = urllib.request.Request(base + "api/open", data=json.dumps(dict(what, agent=opener())).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return base + json.loads(r.read())["page"]
    except urllib.error.HTTPError as e:
        try:
            why = json.loads(e.read()).get("error")
        except ValueError:
            why = str(e)
        raise RuntimeError(why)


def _tern(*args):
    exe = shutil.which("tern")
    if exe is None:
        raise RuntimeError("tern is not on the PATH")
    r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"tern {args[0]}: {(r.stderr or r.stdout).strip()}")
    return r.stdout


def tell(agent, text):
    """Type one line into the agent's Tern pane and press Enter, as if the person had sent it there."""
    window = ["--window", agent["window"]] if agent.get("window") else []
    _tern("send", *window, str(agent["pane"]), "paste", " ".join(text.split("\n")))
    _tern("send", *window, str(agent["pane"]), "keys", "Enter")


def _waiting_file(review):
    key = hashlib.sha1(str(Path(review).expanduser().resolve()).encode()).hexdigest()[:20]
    return S.cache() / "waiting" / f"{key}.json"


@contextlib.contextmanager
def waiting(review):
    """While `mk review open --wait` waits for `review`, the server knows (a Send then tells no pane: --wait has it)."""
    f = _waiting_file(review)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"pid": os.getpid(), "review": str(review)}), encoding="utf-8")
    try:
        yield
    finally:
        f.unlink(missing_ok=True)


def is_waiting(review):
    """Whether a live `mk review open --wait` waits for `review`."""
    try:
        pid = int(json.loads(_waiting_file(review).read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def show(url, where="right", replace=None):
    """Open `url`: in Tern beside $TERN_PANE (`where`: right, down or tab), or in place of the pane `replace`; else in
    the system's browser. Returns {"tern": block} or {"browser": url}."""
    pane = replace or os.environ.get("TERN_PANE", "")
    if shutil.which("tern") and str(pane).isdigit():
        got = json.loads(_tern("browser", json.dumps({"op": "open", "owner": int(pane), "url": url})))
        block = (got.get("ok") or {}).get("block")
        if block is None:
            raise RuntimeError(f"tern browser open: {got.get('error')}")
        if replace:
            _tern("dock", str(block))
            _tern("close", str(replace))
        elif where == "tab":
            _tern("dock", str(block))
            _tern("move", str(block), "new-tab")
        else:
            _tern("dock", str(block), where)
        return {"tern": block}
    webbrowser.open(url)
    return {"browser": url}
