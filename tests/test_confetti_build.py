"""library:confetti in a real `mk build` (the Blender half: the drivers on the pieces, and the keys stage): `start` keyed on
a lane of the [audio] hits file is dealt round the volleys' starts (start0 .. start2, CONSTANT, keyframe i into volley
i % 3), every piece is where core.confetti.pose puts it (position, scale, turn) on a moved and scaled prop, and a lane
`values` key steps a property through its list on the hits; a bad `values` or colour stops the build saying what to
change. Skipped without Blender."""
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN
from mkmmd.core import confetti as CF
from mkmmd.core import shell as S


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

AT, SCALE = (0.4, -0.3, 1.2), 1.5
SLOTS = {"count": 6, "volleys": 3, "seed": 4, "floor": -0.5, "life": 1.6, "reach": 1.2}
HITS = {"accent": [0.2, 0.7, 1.3, 1.9, 2.4], "chug": [0.1, 0.5, 0.9, 1.4]}
SPIN = [1.0, 2.5, 0.5]                           # the `values` the chug steps `spin` through
HEAD = """[project]
name = "t"
fps = 30
frame0 = 31
duration = 3.0
blend = "build/t.blend"

[audio]
hits = "audio/hits.json"
"""
PROJECT = HEAD + f"""
[[prop]]
name = "cf"
card = "library:confetti"
at = [{AT[0]}, {AT[1]}, {AT[2]}]
scale = {SCALE}
slots = {{ count = 6, volleys = 3, seed = 4, floor = -0.5, life = 1.6, reach = 1.2 }}

[[key]]
target = "cf"
prop = "start"
lane = "accent"

[[key]]
target = "cf"
prop = "spin"
lane = "chug"
values = {SPIN}
"""
TS = (0.1, 0.3, 0.8, 1.2, 1.6, 2.0, 2.6)           # clip seconds on whole frames, none on a hit


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


def build(root, text, until="keys"):
    """`mk build` of the project `text` in `root` (with the hits file): (exit code, its JSON report)."""
    (root / "mk.toml").write_text(text, encoding="utf-8")
    (root / "audio").mkdir(exist_ok=True)
    (root / "audio" / "hits.json").write_text(json.dumps(HITS), encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as e:
        MAIN.main(["build", "--project", str(root), "--until", until])
    return e.value.code, json.loads(out.getvalue())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("confetti")
    code, out = build(root, PROJECT)
    assert code == 0, out.get("error")
    return root, out


def query(root, expr, frames, capsys):
    capsys.readouterr()
    code, q = cli(["q", str(root / "build" / "t.blend"), expr, "--frames", ",".join(map(str, frames)), "--project",
                   str(root)], capsys)
    assert code == 0, q
    return q["values"]


def frame(t):
    f = 31 + 30 * t
    assert abs(f - round(f)) < 1e-9, t
    return round(f)


def dealt(t, k):
    """start<k> at clip second t: the accents' keys dealt in turn (CONSTANT; before its first key, that key's value)."""
    mine = HITS["accent"][k::3]
    return max([h for h in mine if h <= t + 1e-9] or mine[:1])


def stepped(t):
    """`spin` at clip second t: SPIN in turn on the chug's hits (before the first, the first value)."""
    n = sum(h <= t + 1e-9 for h in HITS["chug"])
    return SPIN[(max(n, 1) - 1) % len(SPIN)]


def test_the_build_logs_no_warning_and_reports_the_deal(built):
    _, out = built
    assert not [ln for ln in out["log"] if "WARNING" in ln]
    keys = out["stages"]["keys"]
    assert (keys["cf.start"], keys["cf.start0"], keys["cf.start1"], keys["cf.start2"]) == (5, 2, 2, 1)
    assert keys["cf.spin"] == len(HITS["chug"])
    assert out["stages"]["props"]["cf"]["card"] == "library:confetti"


def test_start_keyed_on_a_lane_is_dealt_round_the_volleys(built, capsys):
    root, _ = built
    ts = (0.0, 0.2, 0.5, 1.0, 1.5, 1.9, 2.0, 2.5)
    got = query(root, '[bpy.data.objects["cf"][k] for k in ("start0", "start1", "start2")]', [frame(t) for t in ts],
                capsys)
    for t, starts in zip(ts, got):
        assert starts == pytest.approx([dealt(t, k) for k in range(3)], abs=1e-6), t


def test_a_lane_values_key_steps_a_property_through_its_list(built, capsys):
    root, _ = built
    ts = (0.0, 0.3, 0.5, 0.6, 1.0, 1.5, 2.5)
    got = query(root, 'bpy.data.objects["cf"]["spin"]', [frame(t) for t in ts], capsys)
    assert got == pytest.approx([stepped(t) for t in ts])
    assert got == pytest.approx([1.0, 1.0, 2.5, 2.5, 0.5, 1.0, 1.0])                 # 1.0, 2.5, 0.5, then round again


def test_every_piece_is_where_the_core_puts_it_on_a_moved_scaled_prop(built, capsys):
    root, _ = built
    O = CF.options(SLOTS)
    pieces = CF.layout(O)
    names, at = [], {}
    for p in pieces:
        names.append(f"cf_{p['kind']}{at.get(p['kind'], 0)}")
        at[p["kind"]] = at.get(p["kind"], 0) + 1
    assert {p["kind"] for p in pieces} == set(CF.KINDS)
    expr = ("[[list(m.translation), list(m.to_scale()), [list(r) for r in m.to_3x3().normalized()]] for m in "
            f"(bpy.data.objects[n].matrix_world for n in {names!r})]")
    got = query(root, expr, [frame(t) for t in TS], capsys)
    base = CF.root_values(O)
    flying = 0
    for t, objs in zip(TS, got):
        P = dict(base, spin=stepped(t), **{f"start{k}": dealt(t, k) for k in range(3)})
        for (pos_w, scale_w, R), p, n in zip(objs, pieces, names):
            pos, rot, scale = CF.pose(t, P, p)
            assert pos_w == pytest.approx(list(np.array(AT) + SCALE * pos), abs=2e-5), (t, n)
            assert scale_w == pytest.approx(list(SCALE * scale), abs=2e-6), (t, n)
            assert np.array(R) == pytest.approx(S.rot_matrix(*np.degrees(rot)), abs=2e-4), (t, n)
            flying += scale[0] > 0.5 * P["size"] * p["share"]
    assert flying > 20                                                     # not just hidden pieces
    first = got[0]
    assert all(s == pytest.approx([SCALE * CF.MIN_SCALE] * 3, abs=1e-7) for _, s, _ in first)   # nothing before 0.2


@pytest.mark.parametrize("body, frag", [
    ('[[key]]\ntarget = "cf"\nprop = "spin"\nlane = "chug"\nvalues = [1.0, 2.0]\npulse = [[0.0, 1.0]]\n',
     "give `values` or `pulse`, not both"),
    ('[[key]]\ntarget = "cf"\nprop = "spin"\nlane = "chug"\nvalues = []\n', "values = [v0, v1, ...]"),
    ('[[key]]\ntarget = "cf"\nprop = "spin"\nlane = "chug"\nvalues = ["fast"]\n', "values = [v0, v1, ...]"),
    ('[[key]]\ntarget = "cf"\nprop = "spin"\nkeys = [[0.0, 1.0]]\nvalues = [1.0]\n', "on the hits of a `lane`"),
])
def test_a_bad_values_key_stops_the_build_saying_what_to_change(tmp_path, body, frag):
    code, out = build(tmp_path, HEAD + '\n[[prop]]\nname = "cf"\ncard = "library:confetti"\nslots = { count = 2 }\n\n'
                      + body)
    assert code != 0 and frag in out["error"]


def test_a_bad_colour_stops_the_build_with_the_colour_named(tmp_path):
    code, out = build(tmp_path, HEAD + '\n[[prop]]\nname = "cf"\ncard = "library:confetti"\n'
                      'slots = { count = 2, colors = ["love", "chartreuse"] }\n', until="props")
    assert code != 0 and "confetti: colors[1]" in out["error"]
