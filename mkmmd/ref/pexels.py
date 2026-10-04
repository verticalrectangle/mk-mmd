"""Pexels video API: search, clip metadata, rendition choice, capped downloads.

The API key comes from the system keyring (`secret-tool lookup service pexels key api`). It is sent only to
api.pexels.com as the Authorization header; it is never printed, logged, stored or put in an error message. Video
files come from videos.pexels.com without it."""
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import RefError, RefUsage

API = "https://api.pexels.com"
UA = "mk-mmd-ref/1.0"
LICENSE = "Pexels License"
LICENSE_URL = "https://www.pexels.com/license/"
MAX_DOWNLOAD_BYTES = 150_000_000        # per clip
MIN_FREE_BYTES = 1_500_000_000          # keep this much disk free after a download
RETRY_CODES = (429, 500, 502, 503, 504)


def api_key():
    """The Pexels API key from the keyring."""
    try:
        r = subprocess.run(["secret-tool", "lookup", "service", "pexels", "key", "api"], capture_output=True, text=True,
                           timeout=20)
    except FileNotFoundError:
        raise RefError("secret-tool is not installed, so the Pexels API key cannot be read from the keyring "
                       "(libsecret-tools)")
    except subprocess.TimeoutExpired:
        raise RefError("the keyring did not answer within 20 s")
    key = r.stdout.strip()
    if not key:
        raise RefError("no Pexels API key in the keyring; store it with: "
                       "secret-tool store --label=Pexels service pexels key api")
    return key


def _fetch(url, headers=None, tries=4, timeout=60):
    """GET with retries on rate limits and server errors; returns an open response. Errors never carry headers."""
    where = url.split("?")[0]
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 401:
                raise RefError("Pexels rejected the API key (HTTP 401)")
            if e.code == 404:
                raise RefUsage(f"not found: {where}")
            if e.code not in RETRY_CODES:
                break
            wait = e.headers.get("Retry-After") if e.headers else None
            time.sleep(min(float(wait), 30.0) if wait and wait.isdigit() else 1.5 * (i + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = type(e).__name__
            time.sleep(1.5 * (i + 1))
    raise RefError(f"GET failed ({last}): {where}")


def api_get(path, params=None, key=None):
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    with _fetch(url, {"Authorization": key or api_key()}) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------------------------- renditions
def mp4_files(video):
    return [f for f in video.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("width")
            and f.get("height") and f.get("link")]


def short_side(f):
    return min(f["width"], f["height"])


def pick_rendition(files, min_height=720, max_bytes=MAX_DOWNLOAD_BYTES):
    """The smallest rendition whose short side is >= min_height ('720p' means 1280x720 and 720x1280 alike) and whose
    size fits the cap; at equal resolution the higher frame rate wins (blinks last ~5 frames at 30 fps), capped at 60.
    None when no rendition qualifies."""
    ok = [f for f in files if short_side(f) >= min_height and (f.get("size") or 0) <= max_bytes]
    if not ok:
        return None
    return min(ok, key=lambda f: (short_side(f), f["width"] * f["height"], -min(float(f.get("fps") or 0), 60.0),
                                  f.get("size") or 0))


def describe_rendition(f):
    """'1280x720@25 2.9MB' (compact; a `renditions` list stays readable)."""
    mb = f"{f['size'] / 1e6:.1f}MB" if f.get("size") else "?MB"
    return f"{f['width']}x{f['height']}@{float(f.get('fps') or 0):g} {mb}"


def stored_files(video):
    """The renditions kept in clips.json (no api key involved in the links)."""
    return [{"quality": f.get("quality"), "width": f["width"], "height": f["height"], "fps": f.get("fps"),
             "size": f.get("size"), "link": f["link"]}
            for f in sorted(mp4_files(video), key=lambda f: (short_side(f), f["width"], f.get("fps") or 0))]


def clip_entry(video):
    """The clips.json entry for a Pexels video record."""
    user = video.get("user") or {}
    return {"id": int(video["id"]), "source": "pexels", "url": video.get("url"), "author": user.get("name"),
            "author_url": user.get("url"), "license": LICENSE, "license_url": LICENSE_URL,
            "duration_s": video.get("duration"), "width": video.get("width"), "height": video.get("height"),
            "files": stored_files(video)}


def summarize(video, min_height=720, max_bytes=MAX_DOWNLOAD_BYTES):
    """One search result: what `mk ref search` prints. `renditions` lists the files with a short side of at least 3/4 of
    min_height (thumbnails-sized ones are of no use for tracking and only make the listing long)."""
    files = sorted(mp4_files(video), key=lambda f: (short_side(f), f["width"], f.get("fps") or 0))
    pick = pick_rendition(files, min_height, max_bytes)
    user = video.get("user") or {}
    return {"id": int(video["id"]), "duration": video.get("duration"), "size": [video.get("width"), video.get("height")],
            "fps": float((pick or files[-1]).get("fps") or 0) if files else None,
            "renditions": [describe_rendition(f) for f in files if short_side(f) >= 0.75 * min_height],
            "pick": describe_rendition(pick) if pick else None,
            "url": video.get("url"), "user": user.get("name"), "user_url": user.get("url"), "license": LICENSE}


# ---------------------------------------------------------------------------------------------- API calls
def search(query, n=30, min_height=720, orientation="landscape", max_bytes=MAX_DOWNLOAD_BYTES, key=None):
    """Up to n results that have a usable rendition (see pick_rendition), in the API's relevance order."""
    if not query.strip():
        raise RefUsage("empty search query")
    if orientation not in ("landscape", "portrait", "any"):
        raise RefUsage(f"orientation must be landscape, portrait or any (got {orientation!r})")
    key = key or api_key()
    out, seen, page = [], set(), 1
    while len(out) < n and page <= 10:
        params = {"query": query, "per_page": min(80, max(15, n)), "page": page}
        if orientation != "any":
            params["orientation"] = orientation
        data = api_get("/videos/search", params, key)
        for v in data.get("videos", []):
            if v["id"] in seen:
                continue
            seen.add(v["id"])
            s = summarize(v, min_height, max_bytes)
            if s["pick"]:
                out.append(s)
                if len(out) >= n:
                    break
        if not data.get("next_page") or not data.get("videos"):
            break
        page += 1
    return out


def video(cid, key=None):
    """The Pexels record of one video."""
    try:
        return api_get(f"/videos/videos/{int(cid)}", None, key)
    except RefUsage:
        raise RefUsage(f"no Pexels video with id {cid}")


# ---------------------------------------------------------------------------------------------- download
def check_disk(dest, need_bytes):
    free = shutil.disk_usage(Path(dest).parent if Path(dest).parent.exists() else Path.home()).free
    if free - need_bytes < MIN_FREE_BYTES:
        raise RefError(f"not enough disk space for the download ({free / 1e9:.1f} GB free, {need_bytes / 1e6:.0f} MB "
                       f"needed, {MIN_FREE_BYTES / 1e9:.1f} GB kept free): `mk ref clean --all`")


def download(url, dest, max_bytes=MAX_DOWNLOAD_BYTES, expected=None, label=None):
    """Stream url to dest (via dest.part). Refuses files above max_bytes (by header or while streaming). Returns the
    size in bytes."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with _fetch(url, timeout=120) as r:
        length = int(r.headers.get("Content-Length") or expected or 0)
        if length > max_bytes:
            raise RefError(f"{label or dest.name}: {length / 1e6:.0f} MB exceeds the {max_bytes / 1e6:.0f} MB cap")
        check_disk(dest, length or 1_000_000)
        n = 0
        try:
            with open(part, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    n += len(chunk)
                    if n > max_bytes:
                        raise RefError(f"{label or dest.name}: download exceeds the {max_bytes / 1e6:.0f} MB cap")
                    fh.write(chunk)
            if length and n != length:
                raise RefError(f"{label or dest.name}: incomplete download ({n} of {length} bytes)")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    part.replace(dest)
    print(f"  downloaded {label or dest.name}: {n / 1e6:.1f} MB", file=sys.stderr, flush=True)
    return n
