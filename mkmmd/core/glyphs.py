"""Which characters a font file draws itself (docs/design.md: Text, Fonts), bpy-free: the character map (`cmap`) of a
TrueType or OpenType font, or of a collection's first font.

Blender draws a character its font lacks in a fallback face of its own (String to Curves takes it from the fonts Blender
ships: Noto Sans CJK, Noto Emoji, a last-resort box), so measuring a string can never tell a missing glyph: a katakana
call set in a Latin-only display face comes out in another face, and a kana-only face's "E" is Blender's. The text stage
asks this module instead: it warns about the characters a font lacks and takes the cap height from a reference the font
really has (CAP_REFS)."""
import struct
import unicodedata

CAP_REFS = ("E", "H", "\u30ed", "\u53e3")       # what a cap height is measured on: a Latin capital, else katakana RO, else
#                                               the kanji for mouth (flat-topped: the full height of a kana or kanji face)
SFNT = (b"\x00\x01\x00\x00", b"OTTO", b"true")  # the versions of a TrueType / OpenType font's offset table
INVISIBLE = ("Cc", "Cf")                        # control and format characters (a zero-width joiner): never drawn
SELECTORS = ((0xFE00, 0xFE0F), (0xE0100, 0xE01EF))   # variation selectors (combining marks, not format characters):
#                                               String to Curves draws nothing for them and gives them no width


class FontError(ValueError):
    """A font file whose character map cannot be read."""


def _u16(b, o):
    return struct.unpack_from(">H", b, o)[0]


def _u32(b, o):
    return struct.unpack_from(">I", b, o)[0]


def _format4(b, o, out):
    """Segment mapping to delta values (the BMP): a code in a segment has a glyph when its delta-shifted code, or the glyph
    its range offset points at, is not glyph 0; the closing 0xFFFF segment maps nothing."""
    seg2 = _u16(b, o + 6)
    ends, starts = o + 14, o + 16 + seg2
    deltas, ranges = starts + seg2, starts + 2 * seg2
    for i in range(seg2 // 2):
        end, start = _u16(b, ends + 2 * i), _u16(b, starts + 2 * i)
        delta, ro = _u16(b, deltas + 2 * i), _u16(b, ranges + 2 * i)
        for c in range(start, min(end, 0xFFFE) + 1):
            if ro:
                g = _u16(b, ranges + 2 * i + ro + 2 * (c - start))
                g = (g + delta) & 0xFFFF if g else 0
            else:
                g = (c + delta) & 0xFFFF
            if g:
                out.add(c)


def _format12(b, o, out, many):
    """Segmented coverage (format 12: each code of a group on its own glyph, counting up from the group's first) or
    many-to-one ranges (format 13: the whole group on one glyph)."""
    for i in range(_u32(b, o + 12)):
        g = o + 16 + 12 * i
        start, end, gid = _u32(b, g), min(_u32(b, g + 4), 0x10FFFF), _u32(b, g + 8)
        if many:
            if gid:
                out.update(range(start, end + 1))
        else:
            out.update(range(start + (gid == 0), end + 1))


def read_cmap(b):
    """The code points the font in the bytes `b` maps to a glyph (a frozenset), from every Unicode subtable of its `cmap`
    (platform 0, or Windows' encodings 1 and 10; formats 4, 12 and 13); None when `b` is not a TrueType / OpenType font
    or collection (a Type 1 font, a WOFF) or has no Unicode map it can read. A FontError says the map is cut short."""
    try:
        base = _u32(b, 12) if b[:4] == b"ttcf" else 0          # a collection: its first font
        if b[base:base + 4] not in SFNT:
            return None
        cmap = None
        for i in range(_u16(b, base + 4)):
            r = base + 12 + 16 * i
            if b[r:r + 4] == b"cmap":
                cmap = _u32(b, r + 8)                           # offsets are from the start of the file, in a collection too
        if cmap is None:
            return None
        out, read = set(), False
        for i in range(_u16(b, cmap + 2)):
            r = cmap + 4 + 8 * i
            platform, encoding, o = _u16(b, r), _u16(b, r + 2), cmap + _u32(b, r + 4)
            if not (platform == 0 or (platform == 3 and encoding in (1, 10))):
                continue
            fmt = _u16(b, o)
            if fmt == 4:
                _format4(b, o, out)
            elif fmt in (12, 13):
                _format12(b, o, out, fmt == 13)
            else:
                continue
            read = True
        return frozenset(out) if read else None
    except struct.error:
        raise FontError("the font's character map is cut short") from None


def codepoints(path):
    """read_cmap of the font file at `path`."""
    try:
        with open(path, "rb") as fh:
            b = fh.read()
    except OSError as e:
        raise FontError(f"cannot read {path}: {e.strerror}") from None
    try:
        return read_cmap(b)
    except FontError as e:
        raise FontError(f"{path}: {e}") from None


def missing(cover, s):
    """The characters of `s` a font with the code points `cover` does not draw itself, each once and in order, leaving out
    spaces, line breaks and what is never drawn (control and format characters, variation selectors); [] when the cover
    is not known (None)."""
    if cover is None:
        return []
    return [c for c in dict.fromkeys(s) if not c.isspace() and unicodedata.category(c) not in INVISIBLE
            and not any(a <= ord(c) <= b for a, b in SELECTORS) and ord(c) not in cover]


def cap_refs(cover):
    """The characters a cap height may be measured on, in order of preference: those of CAP_REFS the font has (the Latin
    ones, E and H, when its map is not known)."""
    if cover is None:
        return CAP_REFS[:2]
    return tuple(c for c in CAP_REFS if ord(c) in cover)
