"""Tiny fonts built byte by byte for the tests, with struct alone: the sfnt container and a collection of them, `cmap`
subtables of formats 4, 12 and 13, and a whole TrueType font that FreeType (Blender's String to Curves) loads and draws:
head, hhea, maxp, hmtx, cmap, loca, glyf, name, post and OS/2, its glyphs straight-sided outlines of known size."""
import struct


def _pad4(b):
    return b + b"\0" * (-len(b) % 4)


def _checksum(b):
    b = _pad4(b)
    return sum(struct.unpack(f">{len(b) // 4}I", b)) & 0xFFFFFFFF


def _search(n, unit):
    """(searchRange, entrySelector, rangeShift) of a binary-search header over `n` entries of `unit` bytes."""
    es = max(k for k in range(17) if 2 ** k <= max(n, 1))
    return unit * 2 ** es, es, unit * n - unit * 2 ** es


def sfnt(tables, version=b"\x00\x01\x00\x00", base=0):
    """A font: the offset table, the table records (sorted by tag) and `tables` ({tag: bytes}) after them, each padded to
    4 bytes. The records count from `base` bytes before the font (a font in a collection: from the start of the file)."""
    tags = sorted(tables)
    head = version + struct.pack(">HHHH", len(tags), *_search(len(tags), 16))
    off = len(head) + 16 * len(tags)
    recs, data = b"", b""
    for tag in tags:
        t = tables[tag]
        recs += struct.pack(">4sIII", tag.encode("latin-1"), _checksum(t), base + off + len(data), len(t))
        data += _pad4(t)
    return head + recs + data


def collection(fonts):
    """A TrueType collection (`ttcf`) of the fonts given as {tag: bytes} tables, laid out one after the other."""
    pos = 12 + 4 * len(fonts)
    offs, blobs = [], []
    for tables in fonts:
        offs.append(pos)
        blobs.append(sfnt(tables, base=pos))
        pos += len(blobs[-1])
    return b"ttcf" + struct.pack(f">II{len(fonts)}I", 0x00010000, len(fonts), *offs) + b"".join(blobs)


def cmap(*subtables):
    """A `cmap` table of (platform, encoding, subtable bytes) records."""
    off = 4 + 8 * len(subtables)
    recs, data = b"", b""
    for platform, encoding, sub in subtables:
        recs += struct.pack(">HHI", platform, encoding, off + len(data))
        data += sub
    return struct.pack(">HH", 0, len(subtables)) + recs + data


def format4(segments, close=(0xFFFF, 0xFFFF, 1)):
    """A format 4 subtable of `segments`, each (start, end, delta), or (start, end, delta, [glyph id per code]) for a
    segment that maps through the glyph id array (the delta is added to the ids that are not 0), then the `close` segment
    every format 4 map ends with (0xFFFF onto glyph 0)."""
    segs = list(segments) + [close]
    n = len(segs)
    ranges, ids = [], []
    for i, s in enumerate(segs):
        ranges.append(2 * (n - i) + 2 * len(ids) if len(s) > 3 else 0)
        ids += list(s[3]) if len(s) > 3 else []
    body = struct.pack(f">HHHH{n}HH{n}H{n}H{n}H{len(ids)}H", 2 * n, *_search(n, 2), *[s[1] for s in segs], 0,
                       *[s[0] for s in segs], *[s[2] & 0xFFFF for s in segs], *ranges, *ids)
    return struct.pack(">HHH", 4, 6 + len(body), 0) + body


def format12(groups, many=False):
    """A format 12 subtable of groups (first code, last code, first glyph), or format 13 with `many` (the whole group
    on one glyph)."""
    body = b"".join(struct.pack(">III", *g) for g in groups)
    return struct.pack(">HHIII", 13 if many else 12, 0, 16 + len(body), 0, len(groups)) + body


def _glyf(contours):
    """A simple glyph of on-curve points, every coordinate a 16-bit delta; no instructions. Empty: no outline."""
    if not contours:
        return b""
    pts = [p for c in contours for p in c]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ends, k = [], -1
    for c in contours:
        k += len(c)
        ends.append(k)
    dx = [x - px for (x, _), (px, _) in zip(pts, [(0, 0)] + pts[:-1])]
    dy = [y - py for (_, y), (_, py) in zip(pts, [(0, 0)] + pts[:-1])]
    n = len(pts)
    return _pad4(struct.pack(f">hhhhh{len(ends)}HH{n}B{n}h{n}h", len(contours), min(xs), min(ys), max(xs), max(ys),
                             *ends, 0, *([1] * n), *dx, *dy))


def _name(family):
    recs = [(1, family), (2, "Regular"), (3, f"{family} Regular"), (4, f"{family} Regular"),
            (6, family.replace(" ", "") + "-Regular")]
    strings = [s.encode("utf-16-be") for _, s in recs]
    out, off = struct.pack(">HHH", 0, len(recs), 6 + 12 * len(recs)), 0
    for (nid, _), s in zip(recs, strings):
        out += struct.pack(">HHHHHH", 3, 1, 0x409, nid, len(s), off)
        off += len(s)
    return out + b"".join(strings)


def truetype(glyphs, upm=1000, ascender=800, descender=-200, family="mk tiny"):
    """A TrueType font of `glyphs`, {character: (advance, [contour, ...])}: a contour is a list of on-curve (x, y) points in
    font units, outer contours clockwise and holes counter-clockwise; glyph 0 is an empty .notdef. Its character map is a
    Windows Unicode format 4 subtable."""
    chars = sorted(glyphs, key=ord)
    order = [(500, [])] + [glyphs[c] for c in chars]
    data, loca = b"", [0]
    for _, contours in order:
        data += _glyf(contours)
        loca.append(len(data))
    pts = [p for _, cs in order for c in cs for p in c] or [(0, 0)]
    box = (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
    lsb = [min((p[0] for c in cs for p in c), default=0) for _, cs in order]
    n = len(order)
    cps = [ord(c) for c in chars]
    tables = {
        "head": struct.pack(">IIIIHHqqhhhhHHhhh", 0x00010000, 0x00010000, 0, 0x5F0F3CF5, 0x000B, upm, 0, 0, *box, 0, 8, 2,
                            1, 0),
        "hhea": struct.pack(">IhhhHhhhhhhhhhhhH", 0x00010000, ascender, descender, 0, max(a for a, _ in order), min(lsb),
                            0, box[2], 1, 0, 0, 0, 0, 0, 0, 0, n),
        "maxp": struct.pack(">IHHHHHHHHHHHHHH", 0x00010000, n, max(sum(len(c) for c in cs) for _, cs in order),
                            max(len(cs) for _, cs in order), 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0),
        "hmtx": b"".join(struct.pack(">Hh", a, b) for (a, _), b in zip(order, lsb)),
        "cmap": cmap((3, 1, format4([(cp, cp, k + 1 - cp) for k, cp in enumerate(cps) if cp <= 0xFFFF]))),
        "loca": struct.pack(f">{n + 1}I", *loca),
        "glyf": data,
        "name": _name(family),
        "post": struct.pack(">IIhhIIIII", 0x00030000, 0, -100, 50, 0, 0, 0, 0, 0),
        "OS/2": struct.pack(">HhHHHhhhhhhhhhhh10sIIII4sHHHhhhHHIIhhHHH", 4, 500, 400, 5, 0, 650, 700, 0, 140, 650, 700, 0,
                            480, 50, 250, 0, bytes(10), 1, 0, 0, 0, b"NONE", 0x40, min(cps), min(max(cps), 0xFFFF),
                            ascender, descender, 0, ascender, -descender, 0, 0, box[3] // 2, box[3], 0, 32, 1),
    }
    return sfnt(tables)


def rect(x0, y0, x1, y1, hole=False):
    """A rectangle contour: clockwise (an outline), or counter-clockwise (`hole`)."""
    c = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    return c[::-1] if hole else c
