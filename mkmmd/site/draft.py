"""mk post's draft of an output (out/<name>_<output>_draft.mp4) as the scene viewer decodes it with WebCodecs (Tern's
browser block plays no <video>): the H.264 decoder's setup (the codec string and the avcC record from the file's sample
description) and every video sample in decode order as [file offset, size, presentation time in microseconds,
keyframe], from ffprobe; cached in the site cache by the file's size and time."""
import base64
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def path(proj, output):
    return Path(proj.root) / "out" / f"{proj.name}_{output}_draft.mp4"


def _boxes(data, start, end):
    """(type, payload start, end) of the ISO BMFF boxes in data[start:end]."""
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, i)
        head = 8
        if size == 1:
            size, head = struct.unpack_from(">Q", data, i + 8)[0], 16
        elif size == 0:
            size = end - i
        if size < head:
            return
        yield kind, i + head, i + size
        i += size


def avcc(data):
    """The avcC record of the first H.264 sample description in an MP4 (moov/trak/mdia/minf/stbl/stsd/avc1), or None."""
    def find(start, end):
        for kind, a, b in _boxes(data, start, end):
            if kind in CONTAINERS:
                got = find(a, b)
                if got is not None:
                    return got
            elif kind == b"stsd":
                for entry, a2, b2 in _boxes(data, a + 8, b):              # after version, flags and the entry count
                    if entry in (b"avc1", b"avc3"):
                        for child, a3, b3 in _boxes(data, a2 + 78, b2):   # after the visual sample entry's fields
                            if child == b"avcC":
                                return bytes(data[a3:b3])
        return None
    return find(0, len(data))


def index(mp4, cache_dir):
    """{codec, description (base64 avcC), width, height, samples} of an H.264 MP4 (see the module docstring). Raises
    ValueError when ffprobe is missing or the file is not H.264 in an MP4."""
    mp4 = Path(mp4)
    st = mp4.stat()
    key = hashlib.sha1(f"{mp4.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:20]
    out = Path(cache_dir) / "drafts" / f"{key}.json"
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ValueError("ffprobe is not on PATH (it comes with ffmpeg)")
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,width,height:packet=pts_time,pos,size,flags", "-of", "json", str(mp4)],
                       capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(r.stdout)
        stream = d["streams"][0]
    except (ValueError, KeyError, IndexError):
        raise ValueError(f"ffprobe cannot read {mp4.name}: {r.stderr.strip()[-200:]}")
    record = avcc(memoryview(mp4.read_bytes()))
    if stream.get("codec_name") != "h264" or record is None or len(record) < 4:
        raise ValueError(f"{mp4.name} is not H.264 in an MP4")
    samples = [[int(p["pos"]), int(p["size"]), round(float(p["pts_time"]) * 1e6), "K" in p.get("flags", "")]
               for p in d.get("packets", []) if p.get("pos") not in (None, "N/A") and p.get("pts_time") not in (None, "N/A")]
    doc = {"codec": "avc1." + record[1:4].hex(), "description": base64.b64encode(record).decode(),
           "width": int(stream["width"]), "height": int(stream["height"]), "samples": samples}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    return doc
