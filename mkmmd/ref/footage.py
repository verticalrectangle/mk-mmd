"""`mk ref footage`: Pexels videos downloaded into a project as footage for the `footage` frame effect (docs/design.md:
Frame effects, Footage) and described for the asset registry (kind `reference`: author, pages, licence, a credit line),
so `mk assets credits` names them when a project uses them. The API key stays inside mkmmd.ref.pexels: it is never
printed."""
import re
from pathlib import Path

from . import RefError, RefUsage
from . import pexels as PX

FOLDER = "footage"                       # <project>/footage/pexels_<id>.mp4
ORIENTATIONS = ("landscape", "portrait", "square")
LICENSE = f"{PX.LICENSE} ({PX.LICENSE_URL})"
RESTRICTIONS = ("identifiable people may not appear in a bad light or in a way that is offensive; do not sell unaltered "
                "copies; do not imply endorsement by the people or brands shown; do not redistribute or sell it on other "
                "stock photo or wallpaper platforms; not as part of a trademark or business name")


def orientation(w, h):
    return "square" if w == h else "landscape" if w > h else "portrait"


def pick_best(files, min_height=1080, max_bytes=PX.MAX_DOWNLOAD_BYTES):
    """The rendition footage wants: the largest whose short side is at least `min_height` and whose size fits the cap
    (at equal size the higher frame rate, up to 60); None when none qualifies."""
    ok = [f for f in files if PX.short_side(f) >= min_height and (f.get("size") or 0) <= max_bytes]
    if not ok:
        return None
    return max(ok, key=lambda f: (PX.short_side(f), f["width"] * f["height"], min(float(f.get("fps") or 0), 60.0),
                                  -(f.get("size") or 0)))


def title(video):
    """A video's title from its page's address (pexels.com/video/<words>-<id>/), else "Pexels video <id>"."""
    m = re.search(r"/video/([^/]*?)-?(\d+)/?$", video.get("url") or "")
    words = m.group(1).replace("-", " ").strip() if m else ""
    return words[:1].upper() + words[1:] if words else f"Pexels video {int(video['id'])}"


def entry(video, path, rendition):
    """The registry entry of a downloaded Pexels video: slug pexels_<id>, kind reference, its author and their page, the
    video's page, the Pexels License and what it forbids, a credit line."""
    user = video.get("user") or {}
    author, author_url, name = user.get("name"), user.get("url"), title(video)
    by = f", video by {author}" + (f" ({author_url})" if author_url else "") if author else ", video"
    return {"kind": "reference", "slug": f"pexels_{int(video['id'])}", "path": str(Path(path).resolve()), "name": name,
            "author": author, "author_url": author_url, "source_url": video.get("url"), "license": LICENSE,
            "restrictions": RESTRICTIONS, "credit": f"\u201c{name}\u201d{by} on Pexels", "tags": ["footage", "pexels"],
            "extra": {"pexels_id": int(video["id"]), "rendition": PX.describe_rendition(rendition),
                      "size": [rendition["width"], rendition["height"]], "fps": rendition.get("fps"),
                      "duration_s": video.get("duration")}}


def fetch(ids, folder, min_height=1080, orient=None, key=None, max_bytes=PX.MAX_DOWNLOAD_BYTES):
    """Each Pexels video's best rendition (`pick_best`) into `folder`/pexels_<id>.mp4, kept when a file of its size is
    there already. Returns (got, failed): got [{id, path, video, rendition, bytes, downloaded}], failed [{id, error}]
    (a video of another `orient`ation, one without a rendition big enough under the cap, a failed download)."""
    key = key or PX.api_key()
    got, failed = [], []
    for cid in ids:
        try:
            v = PX.video(cid, key)
            w, h = v.get("width"), v.get("height")
            if orient and w and h and orientation(w, h) != orient:
                raise RefUsage(f"it is {orientation(w, h)} ({w}x{h}), not {orient}")
            files = PX.mp4_files(v)
            r = pick_best(files, min_height, max_bytes)
            if r is None:
                have = ", ".join(PX.describe_rendition(f) for f in sorted(files, key=PX.short_side)) or "none"
                raise RefUsage(f"no rendition with a short side of {min_height} px or more under the "
                               f"{max_bytes / 1e6:.0f} MB cap (it has {have})")
            dest = Path(folder) / f"pexels_{int(cid)}.mp4"
            kept = dest.is_file() and bool(r.get("size")) and dest.stat().st_size == r["size"]
            n = dest.stat().st_size if kept else PX.download(r["link"], dest, max_bytes, expected=r.get("size"),
                                                               label=f"pexels {cid}")
            got.append({"id": int(cid), "path": dest, "video": v, "rendition": r, "bytes": n, "downloaded": not kept})
        except (RefUsage, RefError) as e:
            failed.append({"id": cid, "error": str(e)})
    return got, failed
