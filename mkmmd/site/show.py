"""Open a site page: beside the agent's pane in Tern (a browser block docked to its right, below it or in a new tab),
in place of a pane (the plugin opening a review file from Tern's Files pane), or in the system's browser outside Tern."""
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import webbrowser

from . import server as S


def page(**what):
    """The URL of a page for `review=PATH` or `project=DIR`, from the running server (started when none runs)."""
    base = S.ensure()
    req = urllib.request.Request(base + "api/open", data=json.dumps(what).encode(), method="POST",
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
    r = subprocess.run([shutil.which("tern"), *args], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"tern {args[0]}: {(r.stderr or r.stdout).strip()}")
    return r.stdout


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
