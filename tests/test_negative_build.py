"""`negative = true` screen type in a real `mk build --until text` (mkmmd/blender/build/text.py, mkmmd/core/screentype.py): a plain
text and a lyric entry with `screen` and `negative` build, their objects carry `mk_negative` (and `mk_screen`) while an
ordinary screen text carries only `mk_screen`; misuse is refused with the text stage's error, `text '<name>': ...`, naming the
key. A tiny TrueType font is written byte by byte (font_fixtures.py); the timeline holds synthetic filler words. Skipped
without Blender; nothing is rendered."""
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
          "H": (1000, [rect(100, 0, 300, 500), rect(700, 0, 900, 500), rect(100, 200, 900, 300)]),     # the cap height: 500
          "A": (1000, [rect(100, 0, 300, 500), rect(700, 0, 900, 500), rect(100, 400, 900, 500)]),
          "a": (1000, [rect(100, 0, 900, 300), rect(100, 0, 300, 300)])}
HEAD = """[project]
name = "t"
fps = 30
frame0 = 31
duration = 6.0
blend = "build/t.blend"

[look]
palette = "rose-pine-dawn"

[[shot]]
name = "a"
from = 0.0
to = 6.0
at = [0.0, -5.0, 1.0]
look = [0.0, 0.0, 1.0]

[[output]]
name = "16x9"
size = [320, 180]
"""
PROJECT = HEAD + """
[[text]]
name = "neg"
screen = true
negative = true
font = "fonts/tiny.ttf"
size = 0.1
text = "Aa"

[[text]]
name = "ord"
screen = true
font = "fonts/tiny.ttf"
size = 0.1
text = "aA"

[[text]]
name = "ly"
screen = { anchor = "bottom" }
negative = true
font = "fonts/tiny.ttf"
[text.lyrics]
timeline = "audio/timeline.json"
line = 1
style = "none"
to = 5.0
"""
BAD = [
    ("knockout", 'screen = true\nnegative = true\nknockout = true\n', "text 'bad': negative and knockout both reverse type"),
    ("noscreen", 'negative = true\nat = [0.0, 0.0, 1.0]\nfacing = [0.0, -1.0, 0.0]\n',
     "text 'bad': negative = true needs `screen`"),
    ("notbool", 'screen = true\nnegative = "yes"\n', "text 'bad': negative is true or false, not 'yes'"),
    ("backing", 'screen = true\nnegative = true\nbacking = { color = "love" }\n',
     "text 'bad': negative and backing cannot go together"),
    ("blocks", 'screen = true\nnegative = true\nblocks = { land = 1.0 }\n',
     "text 'bad': negative does not apply to blocks"),
]


def write_project(root, body):
    (root / "mk.toml").write_text(body, encoding="utf-8")
    (root / "fonts").mkdir(exist_ok=True)
    (root / "fonts" / "tiny.ttf").write_bytes(truetype(GLYPHS))
    (root / "audio").mkdir(exist_ok=True)
    words = [{"text": w, "start": s, "end": s + 0.3, "voiced_end": s + 0.3} for w, s in zip(("Aa", "aA"), (1.0, 1.4))]
    (root / "audio" / "timeline.json").write_text(json.dumps({"lines": [{"words": words, "start": 1.0, "end": 1.7}]}),
                                                  encoding="utf-8")


def cli(argv, capsys):
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("negative")
    write_project(root, PROJECT)
    r = subprocess.run([sys.executable, "-m", "mkmmd.cli.main", "build", "--project", str(root), "--until", "text"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return root


def test_negative_text_and_a_negative_lyric_entry_build_and_are_marked_for_the_negative_layer(built, capsys):
    expr = ('{o.name: [o.get("mk_screen"), o.get("mk_negative"), o.get("mk_knockout"), list(o.get("mk_show") or [])] '
            'for o in bpy.data.objects if o.name in ("neg", "ord") or o.name.startswith("ly_")}')
    code, q = cli(["q", str(built / "build" / "t.blend"), expr, "--frames", "61", "--project", str(built)], capsys)
    assert code == 0, q
    got, = q["values"]
    assert got["neg"][:3] == [1, 1, None] and got["ord"][:3] == [1, None, None]          # negative or not: the flag is the difference
    words = {k: v for k, v in got.items() if k.startswith("ly_")}
    assert sorted(words) == ["ly_l1w1", "ly_l1w2"]
    assert all(v[:3] == [1, 1, None] and v[3] for v in words.values())                    # each word: screen, negative, shown for a while


@pytest.mark.parametrize("which, body, msg", BAD, ids=[b[0] for b in BAD])
def test_misuse_is_refused_by_the_text_stage_naming_the_key(which, body, msg, tmp_path, capsys):
    write_project(tmp_path, HEAD + '\n[[text]]\nname = "bad"\nfont = "fonts/tiny.ttf"\nsize = 0.1\ntext = "Aa"\n' + body)
    code, out = cli(["build", "--project", str(tmp_path), "--until", "text"], capsys)
    assert code == 3 and msg in out["error"], out.get("error")
