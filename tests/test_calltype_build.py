"""Fan calls and the font check in a real `mk build` (mkmmd/blender/build/text.py, calltype.py, blocktype.py). A tiny
TrueType font is written byte by byte (font_fixtures.py): a few katakana as plain shapes, ロ a ring 500 units tall, and no
E or H. Blender loads it; a text with a character it lacks is a WARNING naming the character (a lyric word's names only
its numbers); the cap height is the font's own ロ, not the E Blender's fallback face draws; names are counted in UTF-8
bytes as Blender cuts them; a calls entry slams its calls onto the loud hits of a lane in both outputs. The calls are idol
fan calls; the timeline holds synthetic filler words. Skipped without Blender."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from font_fixtures import rect, truetype
from mkmmd import config as CFG
from mkmmd.cli import main as MAIN


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

GLYPHS = {" ": (300, []),
          "ロ": (1000, [rect(100, 0, 900, 500), rect(250, 125, 750, 375, hole=True)]),   # the cap height: 500 units
          "ハ": (1000, [rect(150, 0, 350, 450), rect(650, 0, 850, 450)]),
          "イ": (1000, [rect(400, 0, 600, 500)]),
          "オ": (1000, [rect(100, 350, 900, 450), rect(450, 0, 550, 330)]),
          "！": (1000, [rect(450, 150, 550, 500), rect(450, 0, 550, 80)])}
FILL = ["ロハ", "ハヲ"]                  # filler words of the timeline; ヲ is not in the font
WIDE = "ロ" * 16 + "_1"                  # 18 characters, 50 bytes: the most a text's name may take
HEAD = """[project]
name = "t"
fps = 30
frame0 = 31
duration = {duration}
blend = "build/t.blend"

[look]
palette = "rose-pine-dawn"

[[shot]]
name = "a"
from = 0.0
to = {duration}
at = [0.0, -5.0, 1.0]
look = [0.0, 0.0, 1.0]
"""
PROJECT = HEAD.format(duration=6.0) + f"""
[[output]]
name = "16x9"
size = [320, 180]

[[output]]
name = "9x16"
size = [180, 320]

[audio]
hits = "audio/hits.json"

[[text]]
name = "cap"
at = [0.0, 0.0, 1.0]
facing = [0.0, -1.0, 0.0]
font = "fonts/tiny.ttf"
size = 0.25
text = "ロ"

[[text]]
name = "lack"
at = [0.0, 0.0, 1.6]
facing = [0.0, -1.0, 0.0]
font = "fonts/tiny.ttf"
size = 0.1
text = "ロE"

[[text]]
name = "{WIDE}"
at = [0.0, 0.0, 2.0]
facing = [0.0, -1.0, 0.0]
font = "fonts/tiny.ttf"
size = 0.1
text = "ロ"

[[text]]
name = "ly"
at = [0.0, 0.0, 2.4]
facing = [0.0, -1.0, 0.0]
box = [1.0, 0.3]
font = "fonts/tiny.ttf"
[text.lyrics]
timeline = "audio/timeline.json"
line = 1
style = "none"
to = 5.0

[[text]]
name = "cl"
screen = true
font = "fonts/tiny.ttf"
size = 0.08
[text.calls]
words = ["ハイ！", "オイ★"]
lane = "kick"
min_db = -3
style = "slam"
hold = 0.5
tilt = {{ random = 8 }}
backing = [{{ color = "love", torn = 0, slant = 14 }}, {{ color = "gold", torn = 0, slant = -12 }}]
recycle = {{ fade = 3, slide = 0.5 }}
layout = {{ kind = "slots", slots = [{{ at = [-0.25, 0.2] }}, {{ at = [0.25, -0.2] }}] }}
"""
OUTPUTS = ("16x9", "9x16")
LAND = {1: (61, 76), 2: (91, 106), 3: (106, 121)}   # Blender frames: kicks at 1.0, 2.0 and 2.5 s (not the -9 dB one at
#                                                   1.5); c1 gives slot 0 up at its hold's end, c3 takes it later


def write_project(root, body=PROJECT):
    (root / "mk.toml").write_text(body, encoding="utf-8")
    (root / "fonts").mkdir(exist_ok=True)
    (root / "fonts" / "tiny.ttf").write_bytes(truetype(GLYPHS))
    (root / "audio").mkdir(exist_ok=True)
    words = [{"text": w, "start": s, "end": s + 0.3, "voiced_end": s + 0.3} for w, s in zip(FILL, (1.0, 1.4))]
    (root / "audio" / "timeline.json").write_text(json.dumps({"lines": [{"words": words, "start": 1.0, "end": 1.7}]}),
                                                  encoding="utf-8")
    hits = [{"t": 1.0, "db": 0.0}, {"t": 1.5, "db": -9.0}, {"t": 2.0, "db": -1.0}, {"t": 2.5, "db": -2.0}]
    (root / "audio" / "hits.json").write_text(json.dumps({"kick": hits}), encoding="utf-8")


def cli(argv, capsys):
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("calls")
    write_project(root)
    r = subprocess.run([sys.executable, "-m", "mkmmd.cli.main", "build", "--project", str(root), "--until", "text"],
                       capture_output=True, text=True, timeout=600)
    (root / "build.json").write_text(r.stdout, encoding="utf-8")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return root


def report(root):
    out = json.loads((root / "build.json").read_text(encoding="utf-8"))
    return out["stages"]["text"], [ln.split("WARNING ", 1)[1] for ln in out["log"] if "WARNING " in ln], out


def query(root, expr, frames, capsys):
    code, q = cli(["q", str(root / "build" / "t.blend"), expr, "--frames", ",".join(map(str, frames)), "--project",
                   str(root)], capsys)
    assert code == 0, q
    return q["values"]


def test_a_character_the_font_lacks_is_a_warning_that_names_it_and_a_lyric_word_s_names_numbers_only(built):
    text, warns, out = report(built)
    fallback = "; Blender draws it in a fallback face"
    assert f"text lack: the font fonts/tiny.ttf has no glyph for E (U+0045){fallback}" in warns
    assert f"text ly_l1w2: the font fonts/tiny.ttf has no glyph for 1 character(s) of the word (1, 2){fallback}" in warns
    assert text["lack"]["missing_glyphs"] == 1 and text["ly_l1w2"]["missing_glyphs"] == 1
    assert text["ly_l1w2"]["lyric"] == [1, 2] and "widest" not in text["ly_l1w2"]
    for name in ("cap", "ly_l1w1", WIDE):                                       # the font draws all of these itself
        assert "missing_glyphs" not in text[name] and not any(w.startswith(f"text {name}:") for w in warns)
    blob = json.dumps(out, ensure_ascii=False)
    assert not any(w in blob for w in FILL) and "ヲ" not in blob and "U+30F2" not in blob


def test_the_cap_height_is_the_font_s_own_ro_not_a_fallback_e(built, capsys):
    text, _, _ = report(built)
    assert text["cap"]["cap_mm"] == 250.0 and text["cap"]["ink_m"][1] == pytest.approx(0.25, abs=1e-4)
    expr = ('(lambda o: (lambda zs: [min(zs), max(zs)])([(o.matrix_world @ v.co).z for v in '
            'o.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh().vertices]))(bpy.data.objects["cap"])')
    (z0, z1), = query(built, expr, [61], capsys)
    assert z1 - z0 == pytest.approx(0.25, abs=1e-5)                            # the ring as tall as `size`


def test_names_are_counted_in_utf8_bytes_as_blender_cuts_them(built, tmp_path, capsys):
    expr = ('[o.name for o in bpy.data.objects if o.name.startswith("ロ")] + '
            '[g.name for g in bpy.data.node_groups if g.name.startswith("mk_text_ロ")]')
    assert query(built, expr, [61], capsys)[0] == [WIDE, "mk_text_" + WIDE]   # 50 bytes: built whole
    label = '\n[[text]]\nname = "{}"\nat = [0.0, 0.0, 1.0]\nfacing = [0.0, -1.0, 0.0]\nbox = [1.0, 1.0]\ntext = "AB"\n'
    long_name, long_block = "ロ" * 17, "ロ" * 21                               # 17 and 21 characters
    for sub, body, msg in (
            ("name", label.format(long_name),
             f"text '{long_name}': the name takes 51 bytes (UTF-8: a kana or kanji takes 3), over 50"),
            ("blocks", label.format(long_block) + "[text.blocks]\nland = 1.0\n",
             f"text '{long_block}': the block name '{long_block}_c1' takes 66 bytes (UTF-8), over 63: shorten the "
             "entry's name")):
        root = tmp_path / sub
        root.mkdir()
        (root / "mk.toml").write_text(HEAD.format(duration=2.0) + body, encoding="utf-8")
        code, out = cli(["build", "--project", str(root), "--until", "text"], capsys)
        assert code == 3 and msg in out["error"], out.get("error")


def test_a_calls_entry_slams_its_calls_onto_the_loud_hits_of_a_lane_in_each_output(built, capsys):
    text, warns, out = report(built)
    fallback = "; Blender draws it in a fallback face"
    assert any("calls cl 3 call(s) of 2 on 3 hit(s), Blender frames 61..106, gone by 211" in ln for ln in out["log"])
    for o in OUTPUTS:
        assert text[f"cl@{o}"] == {"calls": {"calls": 3, "hits": 3, "words": 2, "style": "slam", "layout": "slots",
                                             "skipped": [], "first_frame": 61, "last_frame": 106, "cut_frame": 211}}
        calls = [text[f"cl_c{k}@{o}"] for k in LAND]
        assert [c["widest"] for c in calls] == ["ハイ！", "オイ★", "ハイ！"]      # calls may be reported: round-robin
        assert [tuple(c["frames"]) for c in calls] == list(LAND.values())
        assert [c.get("missing_glyphs") for c in calls] == [None, 1, None]
        assert f"text cl_c2@{o}: the font fonts/tiny.ttf has no glyph for ★ (U+2605){fallback}" in warns
    expr = ('{o.name: [list(o["mk_show"]), o.get("mk_aspect"), o.get("mk_screen"), "mk_lyric" in o] '
            'for o in bpy.data.objects if o.name.startswith("cl_")}')
    got, = query(built, expr, [61], capsys)
    assert got == {f"cl_c{k}@{o}": [list(f), o, 1, False] for k, f in LAND.items() for o in OUTPUTS}
