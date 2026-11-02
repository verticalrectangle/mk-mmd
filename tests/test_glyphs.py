"""The font coverage reader (mkmmd/core/glyphs.py): the code points a TrueType / OpenType font or a collection's first font
maps to a glyph, read from character maps built byte by byte here (formats 4, 12 and 13), what is not such a font, a map
cut short, and what the text stage asks of it: the characters a font lacks, and what a cap height may be measured on."""
import pytest

from font_fixtures import cmap, collection, format4, format12, rect, sfnt, truetype
from mkmmd.core import glyphs as GL


def font(*subtables, version=b"\x00\x01\x00\x00"):
    return sfnt({"cmap": cmap(*subtables), "head": bytes(54)}, version=version)


def ords(s):
    return frozenset(map(ord, s))


# ------------------------------------------------------------------------------------------------------ the map
def test_format4_maps_delta_and_range_offset_segments_less_the_codes_on_glyph_0():
    sub = format4([(0x41, 0x43, 1 - 0x41),                     # A B C on glyphs 1 2 3
                   (0x61, 0x63, -0x61),                        # a on glyph 0, b c on 1 2
                   (0x30A1, 0x30A4, 10, [5, 0, 7, 9]),          # through the id array: a 0 there stays 0 whatever the delta
                   (0x4E00, 0x4E01, 10, [0xFFF6, 3])])          # an id the delta wraps round to 0
    assert GL.read_cmap(font((3, 1, sub))) == {0x41, 0x42, 0x43, 0x62, 0x63, 0x30A1, 0x30A3, 0x30A4, 0x4E01}


def test_format4_never_counts_the_closing_0xffff_segment():
    sub = format4([(0x41, 0x41, 1 - 0x41)], close=(0xFFFF, 0xFFFF, 2))     # its delta would put 0xFFFF on glyph 1
    assert GL.read_cmap(font((3, 1, sub))) == {0x41}


def test_format12_maps_each_code_of_a_group_and_format13_a_whole_group_on_one_glyph():
    f12 = format12([(0x1F600, 0x1F602, 0),                      # the first code on glyph 0, the next on 1 and 2
                    (0x20000, 0x20001, 5),
                    (0x10FFFE, 0x110005, 9)])                   # past the last code point: cut at U+10FFFF
    assert GL.read_cmap(font((3, 10, f12))) == {0x1F601, 0x1F602, 0x20000, 0x20001, 0x10FFFE, 0x10FFFF}
    f13 = format12([(0x4E00, 0x4E05, 7), (0x5000, 0x5002, 0)], many=True)
    assert GL.read_cmap(font((0, 6, f13))) == set(range(0x4E00, 0x4E06))


def test_a_collection_is_read_from_its_first_font_with_offsets_from_the_start_of_the_file():
    first = {"cmap": cmap((3, 1, format4([(0x41, 0x42, 1 - 0x41)])))}
    second = {"cmap": cmap((3, 1, format4([(0x5A, 0x5A, 1 - 0x5A)])))}
    assert GL.read_cmap(collection([first, second])) == {0x41, 0x42}


def test_every_unicode_subtable_counts_and_nothing_else_does():
    fmt6 = bytes.fromhex("0006000c0000004100010001")                # format 6 (trimmed table): not read
    fmt14 = bytes.fromhex("000e0000000a00000000")                   # variation sequences: not a map of characters
    f = font((0, 3, format4([(0x42, 0x42, 1 - 0x42)])), (0, 5, fmt14), (1, 0, format4([(0x5A, 0x5A, 1 - 0x5A)])),
             (3, 0, format4([(0x59, 0x59, 1 - 0x59)])), (3, 1, format4([(0x41, 0x41, 1 - 0x41)])),
             (3, 10, format12([(0x1F600, 0x1F600, 3)])))
    assert GL.read_cmap(f) == {0x41, 0x42, 0x1F600}                 # Mac Roman and Windows Symbol left out
    assert GL.read_cmap(font((1, 0, format4([(0x5A, 0x5A, 1 - 0x5A)])), (3, 0, format4([(0x59, 0x59, 1)])))) is None
    assert GL.read_cmap(font((3, 1, fmt6))) is None                 # a Unicode map in a format it does not read


def test_opentype_fonts_are_read_and_what_is_no_truetype_or_opentype_font_is_none():
    sub = (3, 1, format4([(0x41, 0x41, 1 - 0x41)]))
    assert GL.read_cmap(font(sub, version=b"OTTO")) == GL.read_cmap(font(sub, version=b"true")) == {0x41}
    assert GL.read_cmap(b"wOF2" + font(sub)[4:]) is None             # a WOFF2 (and a WOFF) is compressed
    assert GL.read_cmap(b"wOFF" + font(sub)[4:]) is None
    assert GL.read_cmap(b"\x80\x01\x00\x00%!PS-AdobeFont-1.0") is None  # a Type 1 font
    assert GL.read_cmap(b"") is None
    assert GL.read_cmap(sfnt({"head": bytes(54)})) is None          # no character map at all


def test_a_map_cut_short_is_a_font_error():
    whole = font((3, 1, format4([(0x41, 0x43, 1 - 0x41), (0x30A1, 0x30A4, 10, [5, 0, 7, 9])])))
    assert GL.read_cmap(whole) == {0x41, 0x42, 0x43, 0x30A1, 0x30A3, 0x30A4}
    end = 12 + 16 * 2 + len(cmap((3, 1, format4([(0x41, 0x43, 1 - 0x41), (0x30A1, 0x30A4, 10, [5, 0, 7, 9])]))))
    for cut in (20, 40, end - 4):                                   # in the table records, the cmap header, the id array
        with pytest.raises(GL.FontError, match="cut short"):
            GL.read_cmap(whole[:cut])


def test_codepoints_reads_a_font_file_and_names_one_it_cannot_read(tmp_path):
    sub = (3, 1, format4([(0x30ED, 0x30ED, 1 - 0x30ED)]))
    (tmp_path / "a.ttf").write_bytes(font(sub))
    assert GL.codepoints(tmp_path / "a.ttf") == {0x30ED}
    (tmp_path / "b.woff2").write_bytes(b"wOF2" + font(sub)[4:])
    assert GL.codepoints(tmp_path / "b.woff2") is None
    with pytest.raises(GL.FontError, match=r"cannot read .*nope\.ttf"):
        GL.codepoints(tmp_path / "nope.ttf")
    (tmp_path / "c.ttf").write_bytes(font(sub)[:30])
    with pytest.raises(GL.FontError, match=r"c\.ttf: the font's character map is cut short"):
        GL.codepoints(tmp_path / "c.ttf")


def test_the_tiny_truetype_font_maps_its_own_characters_and_nothing_else():
    f = truetype({" ": (300, []), "\u30ed": (1000, [rect(100, 0, 900, 500), rect(250, 125, 750, 375, hole=True)])})
    assert GL.read_cmap(f) == {0x20, 0x30ED}


# ------------------------------------------------------------------------------------------------------ what the stage asks
def test_missing_lists_each_character_the_font_lacks_once_in_order_and_leaves_out_what_is_never_drawn():
    cover = ords("ロハイ ")
    assert GL.missing(cover, "ハイ！ロ\nオイ！A") == ["！", "オ", "A"]
    assert GL.missing(cover, "\u3000\t\r\n \u00a0ロ") == []                      # spaces and line breaks
    assert GL.missing(cover, "\x07\u200b\u200d\u2060\u00ad\ufeffロ") == []      # control and format characters
    assert GL.missing(cover, "ロ★\ufe0fハ\ufe0e\U000e0100") == ["★"]           # a variation selector: never drawn
    assert GL.missing(None, "★オ") == []                                          # the font's map is not known


def test_cap_refs_are_e_h_ro_and_kuchi_in_that_order_among_what_the_font_has():
    assert GL.cap_refs(ords("口ロHE")) == ("E", "H", "ロ", "口")
    assert GL.cap_refs(ords("口H")) == ("H", "口")
    assert GL.cap_refs(ords("ロハイ")) == ("ロ",)
    assert GL.cap_refs(frozenset()) == ()
    assert GL.cap_refs(None) == ("E", "H")                        # not known: the Latin ones, as before
