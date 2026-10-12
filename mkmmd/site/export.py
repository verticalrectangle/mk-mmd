"""`mk export web`: a project's scene and music as a folder of static files any web server can host (docs/design.md:
Exporting the page). The page is mk site's scene view (web/js/scene/, its music timeline and its 3D viewer) with the
server's routes swapped for files (web/js/api_static.js as js/api.js); what it reads goes under data/:

  scene.json, music.json   what the `scene` and `music` routes would answer, with the files' URLs and no local paths
  bake.json                the bake's description as mk build wrote it
  bake.mkz                 bake.bin packed (`pack`): every number on a grid fine enough to look the same, stored as its
                           change from the frame before (or from the cable's shape on the frame before), gzipped
  scene.glb.gz             scene.glb with its textures as WebP no larger than `side`, gzipped
  <output>.mp4             mk post's output, H.264 at most `side` px on its longer side, its sound (the page's clock
                           plays the same sound), with draft_<output>.json, the index the Video view decodes it by
  audio.m4a                the clip's sound: from the first video, else the [audio] clip

The page loads bake.mkz and scene.glb.gz only when 3D is first shown. CREDITS.md and licences/<asset>/ (each asset's
readme files) go beside index.html, since the meshes and textures of the scene leave this machine with it."""
import gzip
import io
import json
import posixpath
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

from .. import __version__
from . import api as API
from . import draft as DR
from . import music as MU
from .server import cache

MAGIC = b"MKZ1"
WEB = Path(__file__).parent / "web"
GENERATOR = "mk export web"
QMAX = 2 ** 30                                   # the largest step count a stream may hold (deltas then fit int32)
# grid steps, in glTF units: metres for positions and matrices (their rotation part is unitless), mm for a lens
STEP = {"joints": 5e-5, "nodes": 5e-5, "cameras": 1e-5, "morphs": 1e-4, "exact": 1.0}
CABLE_STEP = [1e-4] * 3 + [1 / 256] * 3          # a by_frame vertex: position, normal
IMPORT = re.compile(r"""(?:^|\n)\s*(?:import|export)\b[^'"]*?["'](\.{1,2}/[^"']+)["']""")


class ExportError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------- bake.bin
def streams(bake):
    """The blocks of bake.bin as the packer stores them (see the module docstring of mkmmd/blender/build/bake.py), sorted
    by offset: {at, rows, cols, step, delta (a row is its change from the row above), ref (index of the stream it is the
    change from), raw (stored as float32)}."""
    F = int(bake["frames"])
    out = []

    def add(at, rows, cols, step, delta=True, ref=None, raw=False):
        out.append({"at": int(at), "rows": int(rows), "cols": int(cols), "step": step, "delta": delta, "ref": ref,
                    "raw": raw})
        return len(out) - 1

    for a in bake.get("armatures", []):
        add(a["offset"], F, len(a["bones"]) * 12, STEP["joints"])
    for n in bake.get("nodes", []):
        add(n["offset"], n["frames"], 12, STEP["nodes"])
    for s in bake.get("shown", []):
        add(s["offset"], F, 1, STEP["exact"])
    for m in bake.get("morphs", []):
        add(m["offset"], F, m["count"], STEP["morphs"])
    for c in bake.get("cameras", []):
        add(c["offset"], F, 15, STEP["cameras"])
    seen = set()
    for o in bake.get("by_frame", []):
        prev = None                                   # the object's geometry before: a cable moves little from it
        for vo, vc, to, tc in o["geoms"]:
            if vo not in seen:
                seen.add(vo)
                same = prev is not None and out[prev]["rows"] == vc
                prev = add(vo, vc, 6, CABLE_STEP, delta=False, ref=prev if same else None)
            if to not in seen:
                seen.add(to)
                add(to, tc, 4, STEP["exact"], delta=False)
    for at in (bake.get("rings") or {}).get("points", {}).values():
        add(at, F, 3, None, raw=True)              # NaN where the ring is not live
    order = sorted(range(len(out)), key=lambda i: out[i]["at"])
    new = {old: k for k, old in enumerate(order)}
    sorted_ = [out[i] for i in order]
    for k, s in enumerate(sorted_):
        if s["ref"] is not None:
            s["ref"] = new[s["ref"]] if new[s["ref"]] < k else None
    return sorted_


def _cover(blocks, total):
    """`blocks` with raw blocks in the gaps up to `total` floats (refs renumbered); overlapping blocks are an error."""
    out, at, index = [], 0, {}
    gap = lambda a, b: {"at": a, "rows": 1, "cols": b - a, "step": None, "delta": False, "ref": None, "raw": True}  # noqa: E731
    for k, s in enumerate(blocks):
        if s["at"] < at:
            raise ExportError(f"bake.bin: the block at float {s['at']} overlaps the one before it")
        if s["at"] > at:
            out.append(gap(at, s["at"]))
        index[k] = len(out)
        out.append({**s, "ref": None if s["ref"] is None else index[s["ref"]]})
        at = s["at"] + s["rows"] * s["cols"]
    if at > total:
        raise ExportError(f"bake.bin holds {total} floats, its description {at}")
    if at < total:
        out.append(gap(at, total))
    return out


def _zigzag_planes(d):
    """int32 values zigzagged to uint32 and split into four byte planes (low byte first)."""
    d = d.astype(np.int32)
    u = ((d.astype(np.int64) << 1) ^ (d.astype(np.int64) >> 31)).astype(np.uint32)
    return np.ascontiguousarray(u.view(np.uint8).reshape(-1, 4).T).tobytes()


def pack(blob, bake):
    """bake.bin (a float32 array) as MKZ1 bytes (web/js/api_static.js unpacks it; `unpack` here does the same)."""
    blob = np.asarray(blob, np.float32)
    blocks = _cover(streams(bake), blob.size)
    head, parts, kept = [], [], {}                 # kept: each stream's steps, for the streams that are changes from it
    for k, s in enumerate(blocks):
        n = s["rows"] * s["cols"]
        x = blob[s["at"]:s["at"] + n]
        q = None
        if not s["raw"] and np.isfinite(x).all():
            step = np.broadcast_to(np.asarray(s["step"], np.float64), (s["cols"],))
            q = np.rint(x.astype(np.float64).reshape(s["rows"], s["cols"]) / step).astype(np.int64)
            if q.size and np.abs(q).max() >= QMAX:
                q = None
        if q is None:
            head.append({"at": s["at"], "rows": s["rows"], "cols": s["cols"], "mode": "raw"})
            parts.append(x.astype("<f4").tobytes())
            continue
        ref = s["ref"] if s["ref"] in kept else None
        kept[k] = q
        if ref is not None:
            d = q - kept[ref]
        elif s["delta"]:
            d = np.diff(q, axis=0, prepend=np.zeros((1, s["cols"]), np.int64))
        else:
            d = q
        step = s["step"] if isinstance(s["step"], list) else float(s["step"])
        head.append({"at": s["at"], "rows": s["rows"], "cols": s["cols"], "mode": "q", "step": step,
                     "delta": bool(s["delta"]) and ref is None, "ref": ref})
        parts.append(_zigzag_planes(d.reshape(-1)))
    header = json.dumps({"floats": int(blob.size), "streams": head}, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<I", len(header)) + header + gzip.compress(b"".join(parts), compresslevel=9, mtime=0)


def unpack(data):
    """MKZ1 bytes back to the float32 array, as the page rebuilds it."""
    if data[:4] != MAGIC:
        raise ExportError("not MKZ1 data")
    n = struct.unpack_from("<I", data, 4)[0]
    head = json.loads(data[8:8 + n])
    body = gzip.decompress(data[8 + n:])
    out, pos, kept = np.zeros(head["floats"], np.float32), 0, {}
    for k, s in enumerate(head["streams"]):
        count = s["rows"] * s["cols"]
        if s["mode"] == "raw":
            out[s["at"]:s["at"] + count] = np.frombuffer(body, "<f4", count, pos)
            pos += 4 * count
            continue
        planes = np.frombuffer(body, np.uint8, 4 * count, pos).reshape(4, count)
        pos += 4 * count
        u = np.ascontiguousarray(planes.T).view(np.uint32).reshape(-1).astype(np.int64)
        v = ((u >> 1) ^ -(u & 1)).reshape(s["rows"], s["cols"])
        if s.get("ref") is not None:
            v = v + kept[s["ref"]]
        elif s.get("delta"):
            v = np.cumsum(v, axis=0)
        kept[k] = v
        step = np.broadcast_to(np.asarray(s["step"], np.float64), (s["cols"],))
        out[s["at"]:s["at"] + count] = (v * step).reshape(-1).astype(np.float32)
    return out


# ---------------------------------------------------------------------------------------------------------- scene.glb
def web_glb(data, side=1024, quality=85):
    """A .glb with each embedded image as WebP no larger than `side` px (kept as it is when that is not smaller)."""
    from PIL import Image
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ExportError("scene.glb is not a .glb")
    jlen = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + jlen])
    at = 20 + jlen
    blen = struct.unpack_from("<I", data, at)[0] if at + 8 <= len(data) else 0
    binary = data[at + 8:at + 8 + blen]
    if len(doc.get("buffers", [])) != 1:
        return data
    images = {im["bufferView"]: im for im in doc.get("images", []) if "bufferView" in im}
    new = bytearray()
    for i, bv in enumerate(doc.get("bufferViews", [])):
        chunk = binary[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
        if i in images:
            img = Image.open(io.BytesIO(chunk))
            img.load()
            if max(img.size) > side:
                k = side / max(img.size)
                img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
            rgba = img.convert("RGBA")
            mode = "RGBA" if rgba.getchannel("A").getextrema()[0] < 255 else "RGB"
            buf = io.BytesIO()
            rgba.convert(mode).save(buf, "WEBP", quality=quality, alpha_quality=100, method=6)
            if buf.tell() < len(chunk):
                chunk = buf.getvalue()
                images[i]["mimeType"] = "image/webp"
        new += b"\0" * (-len(new) % 4)
        bv["byteOffset"] = len(new)
        bv["byteLength"] = len(chunk)
        new += chunk
    new += b"\0" * (-len(new) % 4)
    doc["buffers"][0]["byteLength"] = len(new)
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(new)
    return (struct.pack("<III", magic, version, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(new), b"BIN\0") + bytes(new))


# ---------------------------------------------------------------------------------------------------------- media
def _ffmpeg(args):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise ExportError("ffmpeg: " + (r.stderr.strip()[-400:] or f"exit {r.returncode}"))


def web_video(src, dst, side, crf, fps):
    """H.264 (yuv420p, a keyframe every second), at most `side` px on its longer side, AAC sound, moov first."""
    scale = (f"scale='if(gte(iw,ih),min(iw,{side}),-2)':'if(gte(iw,ih),-2,min(ih,{side}))':flags=lanczos")
    _ffmpeg(["-i", str(src), "-vf", scale, "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-profile:v", "high",
             "-pix_fmt", "yuv420p", "-g", str(round(fps)), "-bf", "2", "-c:a", "aac", "-b:a", "160k", "-map_metadata", "-1",
             "-movflags", "+faststart", str(dst)])


def _has_audio(mp4):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
                        str(mp4)], capture_output=True, text=True, timeout=60)
    return bool(r.stdout.strip())


# ---------------------------------------------------------------------------------------------------------- the page
def page_files():
    """{relative path: bytes} of the page: index.html, its stylesheets and the modules js/export.js imports, with
    js/api_static.js as js/api.js."""
    files = {"index.html": (WEB / "export.html").read_bytes(), "app/site.css": (WEB / "site.css").read_bytes(),
             "app/export.css": (WEB / "export.css").read_bytes()}
    js = WEB / "js"
    todo, seen = ["export.js"], set()
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        src = (js / ("api_static.js" if rel == "api.js" else rel)).read_text(encoding="utf-8")
        files[f"app/js/{rel}"] = src.encode()
        for m in IMPORT.finditer(src):
            todo.append(posixpath.normpath(posixpath.join(posixpath.dirname(rel), m.group(1))))
    return files


def videos(proj, outputs, source):
    """{output: the mp4 to export} for `source` "final" (mk post's output, else its draft), "draft" or "none"."""
    if source == "none":
        return {}
    out = {}
    for name in outputs:
        final = Path(proj.root) / "out" / f"{proj.name}_{name}.mp4"
        draft = DR.path(proj, name)
        pick = (final if final.is_file() else draft) if source == "final" else draft
        if pick.is_file():
            out[name] = pick
    return out


def check_out(out):
    """Refuse to write over a folder this command did not make."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        man = out / "manifest.json"
        try:
            ok = json.loads(man.read_text(encoding="utf-8")).get("generator") == GENERATOR
        except (OSError, ValueError):
            ok = False
        if not ok:
            raise ExportError(f"{out} is not empty and was not written by `{GENERATOR}`; pass another --out")


def export(proj, out, *, outputs=None, video="final", side=1280, crf=23, tex_side=1024, credits="", licences=()):
    """Write the page for `proj` to `out` (see the module docstring); `credits` is the markdown of CREDITS.md and
    `licences` [(asset slug, readme path)]. Returns the manifest (also written as manifest.json)."""
    out = Path(out).expanduser().resolve()
    check_out(out)
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise ExportError("ffmpeg and ffprobe must be on PATH")
    bake_dir = API.bake_dir(proj)
    for name in ("bake.json", "bake.bin", "scene.glb"):
        if not (bake_dir / name).is_file():
            raise ExportError(f"no baked scene in {bake_dir}: `mk build` bakes it (`mk build --bake` a saved scene)")
    bake = json.loads((bake_dir / "bake.json").read_text(encoding="utf-8"))
    names = [o["name"] for o in bake["outputs"]]
    outputs = names if not outputs else outputs
    unknown = [o for o in outputs if o not in names]
    if unknown:
        raise ExportError(f"no output {', '.join(unknown)} in the bake (outputs: {', '.join(names)})")

    work = out.with_name(out.name + ".part")
    shutil.rmtree(work, ignore_errors=True)
    data = work / "data"
    data.mkdir(parents=True)
    notes = []

    (data / "bake.json").write_text(json.dumps(bake, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    blob = np.fromfile(bake_dir / "bake.bin", dtype="<f4")
    (data / "bake.mkz").write_bytes(pack(blob, bake))
    (data / "scene.glb.gz").write_bytes(gzip.compress(web_glb((bake_dir / "scene.glb").read_bytes(), tex_side),
                                                      compresslevel=9, mtime=0))

    drafts = []
    for name, src in videos(proj, outputs, video).items():
        mp4 = data / f"{name}.mp4"
        web_video(src, mp4, side, crf, proj.fps)
        index = DR.index(mp4, cache())
        if len(index["samples"]) != bake["frames"]:
            notes.append(f"{name}: the video has {len(index['samples'])} frames, the bake {bake['frames']}: the Video "
                         "view and the clock part")
        (data / f"draft_{name}.json").write_text(json.dumps({"url": f"data/{name}.mp4", **index}), encoding="utf-8")
        drafts.append(name)
        if src.name.endswith("_draft.mp4") and video == "final":
            notes.append(f"{name}: no final video (out/{proj.name}_{name}.mp4), its draft is exported")

    music = MU.music(proj.root, cache())
    music.pop("root", None)
    music["checks"] = API.check_marks(proj)
    au, sound = music.get("audio"), data / "audio.m4a"
    first = next((data / f"{n}.mp4" for n in drafts if _has_audio(data / f"{n}.mp4")), None)
    if first is not None:
        _ffmpeg(["-i", str(first), "-vn", "-c:a", "copy", str(sound)])
    elif au and au.get("path"):
        _ffmpeg(["-ss", f"{au.get('offset', 0.0):.6f}", "-t", f"{float(proj.duration):.6f}", "-i", au["path"],
                 "-vn", "-c:a", "aac", "-b:a", "160k", str(sound)])
    music["audio"] = ({"url": "data/audio.m4a", "offset": 0.0} if sound.is_file()
                      else {"problem": au["problem"]} if au and au.get("problem") else None)
    (data / "music.json").write_text(json.dumps(music, ensure_ascii=False), encoding="utf-8")

    sizes = {"bin": (data / "bake.mkz").stat().st_size, "glb": (data / "scene.glb.gz").stat().st_size}
    (data / "scene.json").write_text(json.dumps({"json": "data/bake.json", "bin": "data/bake.mkz",
                                                 "glb": "data/scene.glb.gz", "stale": False, "drafts": drafts,
                                                 "sizes": sizes}), encoding="utf-8")
    for rel, b in page_files().items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    if credits:
        (work / "CREDITS.md").write_text(credits, encoding="utf-8")
    for slug, readme in licences:
        dst = work / "licences" / slug / Path(readme).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(readme, dst)

    files = sorted((p.relative_to(work).as_posix(), p.stat().st_size) for p in work.rglob("*") if p.is_file())
    manifest = {"generator": GENERATOR, "mk": __version__, "project": proj.name, "outputs": outputs, "videos": drafts,
                "bytes": sum(s for _, s in files), "scene_bytes": sizes["bin"] + sizes["glb"],
                "files": [{"path": p, "bytes": s} for p, s in files], "notes": notes}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    work.replace(out)
    return {**manifest, "out": str(out)}
