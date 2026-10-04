"""Reference photos from Wikimedia Commons: what a real thing looks like, to model it from (docs/AGENTS.md: Modelling props).

`mk ref photos QUERY` searches the File namespace of commons.wikimedia.org (api.php, no login), keeps what is a photograph
(JPEG / PNG / WebP, wide enough), reads licence, author and page from the file's metadata, downloads thumbnails of a
standard width, writes SOURCES.json beside them and a contact sheet to look at. Nothing but the licensed thumbnails and
that metadata is stored; the photos are visual references only (never textures, never in a render) and the credits stay
in SOURCES.json.

Politeness (Wikimedia's API etiquette): a descriptive User-Agent, one request at a time with a gap between requests,
`maxlag` on API calls, Retry-After honoured on HTTP 429 / 5xx and exponential backoff when it is missing. Thumbnails are
only served in standard widths (others answer HTTP 400), so `width` snaps up to the next one."""
import html
import io
import json
import re
import shutil
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import RefError, RefUsage

API = "https://commons.wikimedia.org/w/api.php"
UA = "mk-mmd-ref/1.0 (modelling reference photos for a personal video tool; Python urllib)"
THUMB_WIDTHS = (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)       # the widths upload.wikimedia.org serves
PHOTO_MIMES = ("image/jpeg", "image/png", "image/webp")
EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
META = ("LicenseShortName", "LicenseUrl", "UsageTerms", "Artist", "Credit", "AttributionRequired", "NonFree",
        "ImageDescription")
GAP = 1.0                          # seconds between two requests
TRIES = 5
RETRY_CODES = (429, 500, 502, 503, 504)
MAX_WAIT = 60.0                    # longest single wait (s)
MAX_THUMB_BYTES = 15_000_000
MIN_FREE_BYTES = 500_000_000       # kept free on the disk after the download
MAX_CANDIDATES = 200               # titles looked at before giving up on finding `n` fitting photos
BATCH = 50                         # titles per imageinfo request (the API's limit)
SOURCES = "SOURCES.json"
SHEET = "contact_sheet.jpg"
USE_NOTE = "visual reference only for modelling; not used as a texture or in renders"


def snap_width(width):
    """The smallest standard thumbnail width that is at least `width` (the largest when it is bigger than all)."""
    return next((w for w in THUMB_WIDTHS if w >= width), THUMB_WIDTHS[-1])


def slug(title):
    stem = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", title)
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()[:80] or "photo"


def plain(text):
    """Metadata values are HTML (a link around the author's name): their text, on one line."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", str(text or "")))).strip()


def ascii_text(text):
    """`text` as plain ASCII for labels drawn with the built-in font (accents dropped, dashes made plain)."""
    text = str(text).replace("\u2014", "-").replace("\u2013", "-")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


# ---------------------------------------------------------------------------------------------- HTTP
class Http:
    """GET with pacing and backoff. `opener(request, timeout=...)` and `sleep(seconds)` are injectable (the tests do)."""

    def __init__(self, gap=GAP, tries=TRIES, opener=None, sleep=None, clock=time.monotonic):
        self.gap, self.tries, self.clock = gap, tries, clock
        self.opener = opener or (lambda req, timeout: urllib.request.urlopen(req, timeout=timeout))
        self.sleep = sleep or (lambda s: time.sleep(s))
        self.last = None
        self.waited = 0.0
        self.retries = 0

    def _pace(self):
        if self.last is not None:
            wait = self.gap - (self.clock() - self.last)
            if wait > 0:
                self._wait(wait)

    def _wait(self, seconds):
        seconds = min(max(seconds, 0.0), MAX_WAIT)
        self.waited += seconds
        self.sleep(seconds)

    def get(self, url, max_bytes=None):
        """The body of `url`. Retries rate limits and server errors (Retry-After, else 2, 4, 8 ... s)."""
        where = url.split("?")[0]
        last = None
        for attempt in range(self.tries):
            self._pace()
            try:
                with self.opener(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                    body = r.read(max_bytes + 1) if max_bytes else r.read()
                self.last = self.clock()
                if max_bytes and len(body) > max_bytes:
                    raise RefError(f"{where}: more than {max_bytes / 1e6:.0f} MB")
                return body
            except urllib.error.HTTPError as e:
                self.last = self.clock()
                last = f"HTTP {e.code}"
                if e.code == 404:
                    raise RefUsage(f"not found: {where}")
                if e.code not in RETRY_CODES:
                    raise RefError(f"GET failed ({last}): {where}")
                after = e.headers.get("Retry-After") if e.headers else None
                self.retries += 1
                self._wait(float(after) if after and after.strip().isdigit() else 2.0 ** (attempt + 1))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                self.last = self.clock()
                last = type(e).__name__
                self.retries += 1
                self._wait(2.0 ** (attempt + 1))
        raise RefError(f"GET failed after {self.tries} tries ({last}): {where}")

    def api(self, **params):
        """One api.php call: the parsed JSON; `maxlag` errors (the servers are busy) are waited out like HTTP 429."""
        query = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        for attempt in range(self.tries):
            data = json.loads(self.get(API + "?" + urllib.parse.urlencode(query)))
            err = data.get("error")
            if not err:
                return data
            if err.get("code") != "maxlag":
                raise RefError(f"Wikimedia API: {err.get('code')}: {err.get('info', '')}")
            self.retries += 1
            self._wait(float(err.get("lag", 0)) + 2.0 ** (attempt + 1))
        raise RefError("Wikimedia API: the servers stayed lagged")


# ---------------------------------------------------------------------------------------------- search
def search_titles(http, query, offset=0, limit=BATCH):
    """(File titles in relevance order, the next offset or None) for bitmap files matching `query`."""
    data = http.api(action="query", list="search", srsearch=f"{query} filetype:bitmap", srnamespace=6, srlimit=limit,
                    sroffset=offset, srprop="")
    hits = data.get("query", {}).get("search", [])
    return [h["title"] for h in hits], data.get("continue", {}).get("sroffset")


def image_info(http, titles, width):
    """{title: imageinfo} for up to BATCH titles: url, size, mime, a thumbnail at `width`, licence metadata."""
    data = http.api(action="query", prop="imageinfo", titles="|".join(titles), iiprop="url|size|mime|extmetadata",
                    iiurlwidth=width, iiextmetadatafilter="|".join(META))
    out = {}
    for page in data.get("query", {}).get("pages", []):
        info = (page.get("imageinfo") or [None])[0]
        if info:
            out[page["title"]] = info
    return out


def describe(title, info, query):
    """The SOURCES.json entry of a file (without `file`: the name it gets on disk): where it is from and under what terms."""
    meta = {k: (v or {}).get("value", "") for k, v in (info.get("extmetadata") or {}).items()}
    return {"title": re.sub(r"^File:", "", title), "page": info.get("descriptionurl", ""),
            "license": plain(meta.get("LicenseShortName")) or plain(meta.get("UsageTerms")),
            "license_url": plain(meta.get("LicenseUrl")),
            "author": plain(meta.get("Artist")) or plain(meta.get("Credit")),
            "attribution_required": str(meta.get("AttributionRequired", "")).lower() == "true",
            "description": plain(meta.get("ImageDescription"))[:200],
            "width": info.get("width"), "height": info.get("height"), "query": query, "use": USE_NOTE}


def refuse(info, min_width):
    """Why a file is not wanted (a key for the report), or None."""
    meta = {k: (v or {}).get("value", "") for k, v in (info.get("extmetadata") or {}).items()}
    if info.get("mime") not in PHOTO_MIMES:
        return "not_a_photo"
    if str(meta.get("NonFree", "")).lower() == "true":
        return "non_free"
    if not (plain(meta.get("LicenseShortName")) or plain(meta.get("UsageTerms"))):
        return "no_licence_data"
    if int(info.get("width") or 0) < min_width:
        return "too_small"
    return None


# ---------------------------------------------------------------------------------------------- the set on disk
def check_disk(dest, need=MAX_THUMB_BYTES):
    where = Path(dest)
    while not where.exists():
        where = where.parent
    free = shutil.disk_usage(where).free
    if free - need < MIN_FREE_BYTES:
        raise RefError(f"not enough free disk for {dest}: {free / 1e6:.0f} MB free, {MIN_FREE_BYTES / 1e6:.0f} MB are kept")


def load_sources(dest):
    path = Path(dest) / SOURCES
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_sources(dest, entries):
    path = Path(dest) / SOURCES
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def store_image(dest, name, body):
    """Check that `body` decodes as an image and write it; returns (width, height)."""
    try:
        with Image.open(io.BytesIO(body)) as im:
            im.verify()
        with Image.open(io.BytesIO(body)) as im:
            size = im.size
    except Exception as e:                                           # PIL raises many kinds for a broken file
        raise RefError(f"{name}: not a readable image ({type(e).__name__})")
    tmp = Path(dest) / (name + ".part")
    tmp.write_bytes(body)
    tmp.replace(Path(dest) / name)
    return size


def _font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def contact_sheet(dest, entries, out=None, cols=4, tile=(320, 240)):
    """One image of every photo of the set (numbered in SOURCES.json order, with licence and author); returns its path."""
    out = Path(out) if out else Path(dest) / SHEET
    font, tw, th, label = _font(12), tile[0], tile[1], 30
    rows = max(1, -(-len(entries) // cols))
    sheet = Image.new("RGB", (cols * (tw + 6) + 6, rows * (th + label + 6) + 6), (35, 33, 54))
    draw = ImageDraw.Draw(sheet)
    for i, e in enumerate(entries):
        x, y = 6 + (i % cols) * (tw + 6), 6 + (i // cols) * (th + label + 6)
        try:
            with Image.open(Path(dest) / e["file"]) as im:
                im = im.convert("RGB")
                im.thumbnail((tw, th), Image.LANCZOS)
                sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
        except OSError:
            draw.text((x + 6, y + 6), "missing", fill=(235, 111, 146), font=font)
        draw.text((x, y + th + 2), f"{i + 1}. {ascii_text(e['title'])[:44]}", fill=(224, 222, 244), font=font)
        draw.text((x, y + th + 16), ascii_text(f"{e.get('license', '')} - {e.get('author', '')}")[:52],
                  fill=(144, 140, 170), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    return str(out)


# ---------------------------------------------------------------------------------------------- the whole job
def fetch(query, dest, n=12, min_width=1000, width=1280, http=None, sheet=True):
    """Search Commons for `query` and download photographs at least `min_width` px wide as thumbnails of `width` into
    `dest` until the set holds `n` of this query (SOURCES.json and a contact sheet beside them; the same query again adds
    nothing, another query adds its own). Returns the report `mk ref photos` prints (no image data)."""
    query = " ".join(str(query).split())
    if not query:
        raise RefUsage("give a search query, e.g. mk ref photos 'Dodge 600 convertible'")
    if n < 1:
        raise RefUsage("--n must be at least 1")
    dest = Path(dest)
    http = http or Http()
    thumb = snap_width(int(width))
    entries = load_sources(dest)
    have = {e["title"] for e in entries}
    already = sum(e.get("query") == query for e in entries)
    want = max(n - already, 0)
    skipped, added, failed, seen = {}, [], [], 0
    offset = 0 if want else None
    while len(added) < want and seen < MAX_CANDIDATES and offset is not None:
        titles, offset = search_titles(http, query, offset)
        if not titles:
            break
        seen += len(titles)
        info = image_info(http, titles, thumb)
        for title in titles:                                        # relevance order
            if len(added) >= want:
                break
            if re.sub(r"^File:", "", title) in have:
                skipped["already_in_set"] = skipped.get("already_in_set", 0) + 1
                continue
            if title not in info:
                skipped["no_image_info"] = skipped.get("no_image_info", 0) + 1
                continue
            why = refuse(info[title], min_width)
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            dest.mkdir(parents=True, exist_ok=True)
            check_disk(dest)
            entry = describe(title, info[title], query)
            ext = EXT[info[title]["mime"]]
            name, k = slug(entry["title"]) + ext, 1
            while (dest / name).exists() or any(e["file"] == name for e in entries):
                k += 1
                name = f"{slug(entry['title'])}_{k}{ext}"
            try:
                body = http.get(info[title].get("thumburl") or info[title]["url"], max_bytes=MAX_THUMB_BYTES)
                size = store_image(dest, name, body)
            except RefError as e:
                failed.append({"title": entry["title"], "error": str(e)})
                continue
            entries.append({"file": name, **entry, "thumb_width": size[0], "thumb_height": size[1]})
            added.append(name)
            have.add(entry["title"])
            save_sources(dest, entries)                              # after every photo: an interrupted run keeps them
    out = {"query": query, "dir": str(dest), "added": len(added), "count": len(entries), "requested": n,
           "min_width": min_width, "width": thumb, "skipped": skipped, "failed": failed,
           "sources": str(dest / SOURCES) if entries else None,
           "photos": [{k: e.get(k) for k in ("file", "title", "page", "license", "author", "width", "height")}
                      for e in entries],
           "http": {"retries": http.retries, "waited_s": round(http.waited, 1)},
           "note": "visual references only: licence and author are in SOURCES.json; look at the contact sheet before "
                   "modelling, and name the angles you will compare against"}
    if int(width) != thumb:
        out["width_note"] = f"{width} px is not a thumbnail width Wikimedia serves: used {thumb}"
    if already:
        out["already_in_set"] = already
    if len(added) < want:
        out["short"] = (f"only {len(added)} of {want} new photos fit (>= {min_width} px wide, a photograph, licence data); "
                        f"try other words or --min-width")
    if entries and sheet:
        out["sheet"] = contact_sheet(dest, entries)
    return out
