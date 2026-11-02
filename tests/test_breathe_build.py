"""A breathing palette in a real `mk build` (mkmmd/blender/build/breathe.py, docs/design.md: Lights and the look): the probe
build with the other palette beside the build, every colour it took from the palette driven by the mix, so that on the
frames of mix 0 and mix 1 every material, world and light value is the plain build's with that palette (the `lights`
factor on top), ramps and images crossfaded, and the cast's own materials left alone. The cast is a mannequin made by
`mk model build` (about a second). Skipped without Blender."""
import json
import os
from pathlib import Path

import pytest

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

HEAD = """[project]
name = "t"
fps = 30
frame0 = 1
duration = 2.0
blend = "build/t.blend"

[scene]
start = 1

[[output]]
name = "16x9"
size = [320, 180]
"""

LOOK = """
[look]
palette = "{palette}"
"""

BREATHE = """
[look.breathe]
palette = "rose-pine-moon"
keys = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
lights = { key = 2.0 }
"""

BODY = """
[[set]]
name = "sky"
kind = "night_sky"

[[set]]
name = "paper"
kind = "sweep"
color = "surface"

[[cast]]
name = "mq"
pmx = "{pmx}"
rig = "{rig}"
at = [-1.0, 0.0, 0.0]

[[prop]]
name = "card"
card = "library:colorbar_card"
at = [0.6, -0.5, 1.2]

[[prop]]
name = "mug"
card = "library:cafe_mug"
at = [0.0, -1.0, 0.0]

[[ribbon]]
name = "hoop"
style = "neon"
color = "iris"
swoop = {{ points = [[-1.0, -1.0, 0.6], [0.0, -1.5, 1.4], [1.0, -1.0, 0.8]] }}
draw = [0.0, 0.5]
clear = false

[[light]]
name = "key"
kind = "area"
color = "gold"
power = {power}
at = [2.0, -3.0, 3.0]
look = [0.0, 0.0, 1.0]

[[light]]
name = "rim"
kind = "point"
color = "love"
power = 200
at = [-2.0, 2.0, 2.5]
keys = [[0.0, 1.0], [2.0, 0.5]]

[[text]]
name = "tag"
at = [0.0, 1.0, 1.8]
facing = [0, -1, 0]
size = 0.2
text = "MILE 7"
color = "pine"
backing = {{ color = "gold" }}

[[shot]]
name = "wide"
from = 0.0
to = 2.0
at = [0.0, -6.0, 1.6]
look = [0.0, 0.0, 1.0]
"""

# every numeric node-socket value of the materials and the world, and every light's colour and strength
SNAP = ('{"sockets": {f"{m.name}|{n.name}|{i}": [round(float(x), 5) for x in (s.default_value if hasattr('
        's.default_value, "__len__") else [s.default_value])] for m in list(bpy.data.materials) + [scene.world] if m is '
        'not None and m.node_tree is not None for n in m.node_tree.nodes for i, s in enumerate(n.inputs) if hasattr(s, '
        '"default_value") and not s.is_linked and not isinstance(s.default_value, (str, bool))}, "lights": {l.name: '
        '[round(c, 5) for c in l.color] + [round(l.energy, 3)] for l in bpy.data.lights}}')
CAST_DRIVERS = ('sum(len(s.material.node_tree.animation_data.drivers) if s.material.node_tree and '
                's.material.node_tree.animation_data else 0 for o in bpy.data.objects if o.name.startswith("Mq") '
                'for s in o.material_slots if s.material)')


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def mannequin(tmp_path_factory):
    d = tmp_path_factory.mktemp("mannequin")
    (d / "m.toml").write_text('[model]\nname = "mq"\nparts = ["mannequin", "mannequin_hair"]\n'
                              f'out = "{d / "out"}"\n[model.needs]\nmannequin = []\n', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        MAIN.main(["model", "build", str(d / "m.toml"), "--no-cache"])
    assert e.value.code == 0
    return d / "out" / "mq.pmx", d / "out" / "mq.rig.json"


@pytest.fixture(scope="module")
def built(tmp_path_factory, mannequin):
    """{"breath" | "dawn" | "moon": project folder}: the breathing project and the plain ones (the key light twice as
    strong in the plain moon one, as the breath's `lights` factor makes it at mix 1)."""
    out = {}
    for name, palette, power, breathe in (("breath", "rose-pine-dawn", 300, True), ("dawn", "rose-pine-dawn", 300, False),
                                          ("moon", "rose-pine-moon", 600, False)):
        d = tmp_path_factory.mktemp(name)
        text = HEAD + LOOK.format(palette=palette) + (BREATHE if breathe else "") + BODY.format(
            pmx=mannequin[0], rig=mannequin[1], power=power)
        (d / "mk.toml").write_text(text, encoding="utf-8")
        with pytest.raises(SystemExit) as e:
            MAIN.main(["build", "--project", str(d), "--skip", "bake", "--skip", "clip"])
        assert e.value.code == 0
        out[name] = d
    return out


def q(capsys, d, expr, frames):
    code, res = cli(["q", "--project", str(d), expr, "--frames", frames], capsys)
    assert code == 0, res
    return res["values"]


def test_the_breathing_build_drives_its_colours_and_reports_them(built, capsys):
    d = built["breath"]
    assert not list((d / ".mk" / "cache" / "breathe").glob("*")), "the probe's folder is left behind"
    mixes = q(capsys, d, 'round(scene["mk_breathe"], 5)', "1,16,31,46,60")    # the clip ends on frame 60 (t 59/30)
    assert mixes == [0.0, 0.5, 1.0, 0.5, pytest.approx(0.00326, abs=1e-5)]
    drivers = q(capsys, d, 'sum(len(i.animation_data.drivers) for c in (bpy.data.materials, bpy.data.lights, '
                           'bpy.data.worlds, bpy.data.node_groups) for x in c for i in (x, getattr(x, "node_tree", None)) '
                           'if i is not None and i.animation_data)', "1")[0]
    assert drivers > 50
    crossfades = q(capsys, d, 'sorted({n.name.split("~")[0] + "~" + n.bl_idname for m in bpy.data.materials if m.node_tree '
                              'for n in m.node_tree.nodes if "~breathe" in n.name and n.bl_idname in ('
                              '"ShaderNodeValToRGB", "ShaderNodeTexImage")})', "1")[0]
    assert any(c.endswith("ShaderNodeValToRGB") for c in crossfades)       # the mug's ramps
    assert any(c.endswith("ShaderNodeTexImage") for c in crossfades)       # the colour bars' picture
    assert q(capsys, d, CAST_DRIVERS, "1") == [0]                          # the cast's own materials left alone


def test_at_mix_0_and_1_every_value_is_the_plain_builds(built, capsys):
    breath = q(capsys, built["breath"], SNAP, "1,31")
    dawn = q(capsys, built["dawn"], SNAP, "1")[0]
    moon = q(capsys, built["moon"], SNAP, "31")[0]
    for got, want, what in ((breath[0], dawn, "mix 0 (dawn)"), (breath[1], moon, "mix 1 (moon)")):
        assert set(want["lights"]) <= set(got["lights"])
        for k, v in want["lights"].items():
            assert got["lights"][k] == pytest.approx(v, rel=1e-4, abs=1e-4), (what, k)
        missing = set(want["sockets"]) - set(got["sockets"])
        assert not missing, (what, sorted(missing)[:5])
        off = {k: (got["sockets"][k], v) for k, v in want["sockets"].items()
               if got["sockets"][k] != pytest.approx(v, rel=1e-4, abs=2e-5)}
        assert not off, (what, sorted(off.items())[:8])
    assert breath[0] != breath[1]                                          # and it did change in between


def test_light_strength_follows_its_factor_and_its_own_keys(built, capsys):
    key = q(capsys, built["breath"], 'round(bpy.data.lights["key"].energy, 3)', "1,16,31")
    assert key == [300.0, pytest.approx(450.0, abs=0.01), 600.0]          # x lerp(1, 2, mix)
    rim = q(capsys, built["breath"], 'round(bpy.data.lights["rim"].energy, 3)', "1,31")
    assert rim == [200.0, pytest.approx(150.0, abs=0.01)]                  # its own [[light]] keys, no factor


def test_a_bad_breathe_table_stops_the_build_before_it_starts(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(HEAD + LOOK.format(palette="rose-pine-dawn")
                                      + '[look.breathe]\npalette = "rose-pine-noon"\nkeys = [[0, 0]]\n', encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path)], capsys)
    assert code != 0 and "[look] breathe: palette = 'rose-pine-noon'" in out["error"]
