"""Block type in a real `mk build` (mkmmd/blender/build/blocktype.py): screen-type lyric blocks built once per output and a
label of blocks standing in the scene, each block a mesh object under its entry's empty, keyed where the core's plan puts
it (scaled to nothing before it drops in), its colours object-linked materials, its numbers in custom properties and its
letters only geometry. Synthetic filler words. Skipped without Blender."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN
from mkmmd.core import blocktype as BT
from mkmmd.core import typeset as T


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

FILL = ["abc", "de", "fghi"]
LABEL = {"name": "lb", "at": [0.0, 0.0, 0.5], "facing": [0.0, -1.0, 0.0], "box": [1.0, 1.0], "text": "AB 12",
         "blocks": {"land": 1.0}}
PROJECT = """[project]
name = "t"
fps = 30
frame0 = 31
duration = 6.0
blend = "build/t.blend"

[[output]]
name = "16x9"
size = [320, 180]

[[output]]
name = "9x16"
size = [180, 320]

[audio]
hits = "audio/hits.json"

[look]
palette = "rose-pine-dawn"

[[shot]]
name = "a"
from = 0.0
to = 6.0
at = [0.0, -5.0, 1.0]
look = [0.0, 0.0, 1.0]

[[text]]
name = "bk"
screen = { anchor = "bottom", height = 0.3 }
[text.lyrics]
timeline = "audio/timeline.json"
line = 1
style = "blocks"
to = 5.0
[text.blocks]
hits = { lane = "kick", move = "slam" }

[[text]]
name = "lb"
at = [0.0, 0.0, 0.5]
facing = [0.0, -1.0, 0.0]
box = [1.0, 1.0]
text = "AB 12"
[text.blocks]
land = 1.0
"""


def write_project(root, body=PROJECT):
    (root / "mk.toml").write_text(body, encoding="utf-8")
    (root / "audio").mkdir(exist_ok=True)
    words = [{"text": w, "start": s, "end": s + 0.3, "voiced_end": s + 0.3} for w, s in zip(FILL, (1.0, 1.4, 1.9))]
    (root / "audio" / "timeline.json").write_text(json.dumps({"lines": [{"words": words, "start": 1.0, "end": 2.2}]}),
                                                  encoding="utf-8")
    (root / "audio" / "hits.json").write_text(json.dumps({"kick": [3.0]}), encoding="utf-8")


def cli(argv, capsys):
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("blocks")
    write_project(root)
    out = root / "build.json"
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "mkmmd.cli.main", "build", "--project", str(root), "--until", "text"],
                       capture_output=True, text=True, timeout=600)
    out.write_text(r.stdout, encoding="utf-8")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return root


def query(root, expr, frames, capsys):
    code, q = cli(["q", str(root / "build" / "t.blend"), expr, "--frames", ",".join(map(str, frames)), "--project",
                   str(root)], capsys)
    assert code == 0, q
    return q["values"]


def test_the_build_reports_each_output_s_blocks_in_numbers_only(built):
    out = json.loads((built / "build.json").read_text())
    text = out["stages"]["text"]
    assert {"bk@16x9", "bk@9x16", "lb"} <= set(text)
    for name in ("bk@16x9", "bk@9x16"):
        s = text[name]
        assert s["blocks"] == 9 and s["words"] == 3 and s["first_frame"] == 31 + 30 and s["moves"] == {"slam": 9}
        assert [w["lyric"] for w in s["per_word"]] == [[1, 1], [1, 2], [1, 3]]
    assert text["lb"]["label"] and text["lb"]["blocks"] == 4
    blob = (built / "build.json").read_text()
    assert not any(re.search(r"\b" + w + r"\b", blob) for w in FILL)


def test_every_block_is_a_mesh_under_its_entry_with_its_numbers_and_screen_flags(built, capsys):
    expr = ('[sorted(o.name for o in bpy.data.objects if o.name.startswith("bk_")), '
            'bpy.data.objects["bk_l1w2c1@9x16"].parent.name, '
            '{k: (list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v) '
            ' for k, v in bpy.data.objects["bk_l1w2c1@9x16"].items()}, '
            'bpy.data.objects["bk_l1w2c1@9x16"].hide_render, bpy.data.objects["lb_c1"].hide_render, '
            'sorted({o.type for o in bpy.data.objects if o.name.startswith(("bk", "lb"))}), '
            '[n.name for n in bpy.data.node_groups if n.name.startswith("mk_block_glyph")], '
            '[s.link for s in bpy.data.objects["lb_c1"].material_slots], '
            'all(s.material is not None for s in bpy.data.objects["lb_c1"].material_slots)]')
    (names, parent, props, hidden, lb_hidden, types, glyph_trees, links, filled), = query(built, expr, [31], capsys)
    assert len(names) == 18 and "bk_l1w3c4@16x9" in names and "bk_l1w3c4@9x16" in names
    assert parent == "bk@9x16" and hidden and not lb_hidden
    assert props["mk_lyric"] == [1, 2] and props["mk_block"] == 1 and props["mk_screen"] == 1
    assert props["mk_aspect"] == "9x16" and props["mk_show"][0] < 31 + 42 <= props["mk_show"][1] <= 31 + 150
    assert types == ["EMPTY", "MESH"] and glyph_trees == []                    # letters are geometry, the reader is gone
    assert set(links) == {"OBJECT"} and len(links) >= 8 and filled


def test_a_label_block_is_keyed_where_the_core_plan_puts_it_and_is_nothing_before_it_drops_in(built, capsys):
    plan = BT.plan(LABEL, fps=30.0, frame0=31, panel=(1.0, 1.0), end=180)
    b = {x.tag: x for x in plan.blocks}["c1"]
    F = T.surface_matrix(LABEL["at"], LABEL["facing"], (0.0, 0.0, 1.0), 0.002) @ plan.matrix
    frames = [31 + b.on - 1, 31 + b.on, 31 + b.land, 31 + b.land + 3, 31 + b.land + 40]
    expr = '[list(bpy.data.objects["lb_c1"].matrix_world.translation), list(bpy.data.objects["lb_c1"].scale)]'
    got = query(built, expr, frames, capsys)
    assert got[0][1] == pytest.approx([0.0, 0.0, 0.0])                          # not there yet
    for f, (pos, scale) in zip(frames[1:], got[1:]):
        loc, _q, s = (b.keys[k][list(b.keys["frames"]).index(f - 31)] for k in ("loc", "quat", "scale"))
        assert pos == pytest.approx((F @ np.array([*loc, 1.0]))[:3], abs=1e-4)
        assert scale == pytest.approx(s, abs=1e-4)
    assert got[2][1][1] < 0.95                                                   # squashed on its landing frame


def test_a_bad_blocks_table_is_a_build_error_that_names_the_key(tmp_path, capsys):
    write_project(tmp_path, PROJECT.replace("land = 1.0", "land = 1.0\ncolour = \"love\""))
    code, out = cli(["build", "--project", str(tmp_path), "--until", "text"], capsys)
    assert code == 3 and "text 'lb': blocks: unknown keys ['colour']" in out["error"]
