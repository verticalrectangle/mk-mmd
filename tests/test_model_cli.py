"""mk model: the command line (no Blender: planning, builders only) and the end-to-end route with Blender when it is
installed (mannequin -> PMX -> mmd_tools import -> verification -> rig.json -> review .blend)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN
from mkmmd.model import build as BD
from mkmmd.model import part as P
from mkmmd.model import spec as SP


def run_cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    out = capsys.readouterr().out
    return e.value.code, json.loads(out)


@pytest.fixture(autouse=True)
def own_cache(tmp_path, monkeypatch):
    """The builds here keep their part cache (and Blender's job folders) in the test's folder, not the user's."""
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))


def mannequin_spec(tmp_path, extra=""):
    p = tmp_path / "m.toml"
    p.write_text('[model]\nname = "mq"\nparts = ["mannequin", "mannequin_hair", "mannequin_skirt"]\n'
                 f'out = "{tmp_path / "out"}"\n[model.needs]\nmannequin = []\n{extra}', encoding="utf-8")
    return p


def test_info_lists_plan_and_needs(tmp_path, capsys):
    code, out = run_cli(["model", "info", str(mannequin_spec(tmp_path))], capsys)
    assert code == 0 and out["parts"] == ["mannequin", "mannequin_hair", "mannequin_skirt"]
    assert out["needs"]["mannequin_hair"] == ["mannequin"] and out["name"] == "mq"
    code, out = run_cli(["model", "info", str(mannequin_spec(tmp_path)), "--only", "mannequin_skirt"], capsys)
    assert out["parts"] == ["mannequin", "mannequin_skirt"]


def test_bad_spec_and_unknown_part_are_usage_errors(tmp_path, capsys):
    code, out = run_cli(["model", "info", str(tmp_path / "missing.toml")], capsys)
    assert code == 2 and "does not exist" in out["error"]
    code, out = run_cli(["model", "info", str(mannequin_spec(tmp_path)), "--only", "hat"], capsys)
    assert code == 2 and "hat" in out["error"]


def test_no_export_builds_and_checks_parts(tmp_path, capsys):
    """--no-export runs and checks the builders in a folder of its own: an exported model beside it (its build.json and
    textures) stays as it was."""
    spec = mannequin_spec(tmp_path)
    export = Path(SP.load(spec)["model"]["out"]).expanduser()
    (export / "tex").mkdir(parents=True)
    (export / "build.json").write_text('{"ok": true, "stage": "done"}', encoding="utf-8")
    code, out = run_cli(["model", "build", str(spec), "--no-export"], capsys)
    assert code == 0 and out["ok"] is True
    names = [p["name"] for p in out["parts"]]
    assert names == ["mannequin", "mannequin_hair", "mannequin_skirt"]
    body = out["parts"][0]
    assert body["bones"] == 76 and body["morphs"] == 6 and body["bodies"] == 19 and body["materials"] == 4
    assert out["textures"] >= 8 and out["warnings"] == []
    assert Path(out["out"]) == (export / "no_export").resolve()
    tex = Path(out["out"]) / "tex"
    assert (tex / "mannequin_skin.png").exists() and not (Path(out["out"]) / "mannequin.pmx").exists()
    assert (export / "build.json").read_text(encoding="utf-8") == '{"ok": true, "stage": "done"}'
    assert not list((export / "tex").iterdir())
    # --only builds into its own folder and rebuilds fewer parts
    code, out = run_cli(["model", "build", str(spec), "--no-export", "--only", "mannequin_hair"], capsys)
    assert code == 0 and [p["name"] for p in out["parts"]] == ["mannequin", "mannequin_hair"]
    assert Path(out["out"]) == (export / "only_mannequin_hair" / "no_export").resolve()


def test_spec_keys_no_part_reads_are_warned(tmp_path, capsys):
    """Keys no part reads (a misspelt one, a table nobody reads) are warned about in the report, one line per top-level
    table, and the keys that are read are not; a partial build checks only the tables of the parts it built."""
    extra = '[mannequin_hair]\nbangs_bones = 4\nlenght = 0.3\n[colors.nothing]\nx = "#102030"\ny = "#203040"\n'
    spec = mannequin_spec(tmp_path, extra)
    code, out = run_cli(["model", "build", str(spec), "--no-export"], capsys)
    assert code == 0 and out["ok"] is True
    warned = [w for w in out["warnings"] if w.startswith("[spec] WARNING")]
    assert len(warned) == 2 and "mannequin_hair.lenght" in warned[0] and "colors.nothing" in warned[1]
    assert "bangs_bones" not in " ".join(warned)
    code, out = run_cli(["model", "build", str(spec), "--no-export", "--only", "mannequin"], capsys)
    assert code == 0 and not [w for w in out["warnings"] if w.startswith("[spec]")]


def test_builder_failure_is_reported_not_raised(tmp_path, capsys):
    saved = dict(BD.REGISTRY)

    @BD.builder("boom_part", needs=())
    def build(ctx):
        raise RuntimeError("kaboom")
    try:
        p = tmp_path / "b.toml"
        p.write_text(f'[model]\nname = "b"\nparts = ["boom_part"]\nout = "{tmp_path / "o"}"\n', encoding="utf-8")
        code, out = run_cli(["model", "build", str(p), "--no-export"], capsys)
        assert code == 1 and out["ok"] is False and out["stage"] == "parts" and "kaboom" in out["error"]
    finally:
        BD.REGISTRY.clear()
        BD.REGISTRY.update(saved)


def test_set_overrides_reach_the_builders(tmp_path, capsys):
    spec = mannequin_spec(tmp_path)
    code, out = run_cli(["model", "build", str(spec), "--no-export", "--set", "mannequin_hair.bangs_bones=5"], capsys)
    hair = next(p for p in out["parts"] if p["name"] == "mannequin_hair")
    assert code == 0 and hair["bones"] == 5


def test_new_starts_a_character_on_a_base(tmp_path, capsys):
    """`mk model new NAME` writes NAME/model.toml: the base under the character's own [model] (name, out), loadable as is;
    it never overwrites a spec, and an unknown base is a usage error naming the bases."""
    d = tmp_path / "mika"
    code, out = run_cli(["model", "new", "mika", "--dir", str(d), "--out", str(tmp_path / "out")], capsys)
    assert code == 0 and out["base"] == "girl" and Path(out["created"]) == (d / "model.toml").resolve()
    s, base = SP.load(d / "model.toml"), SP.load("base:girl")
    assert SP.model_cfg(s)["name"] == "mika" and SP.model_cfg(s)["out"] == tmp_path / "out"
    assert {k: v for k, v in s.items() if k != "model"} == {k: v for k, v in base.items() if k != "model"}
    code, out = run_cli(["model", "new", "mika", "--dir", str(d)], capsys)
    assert code == 2 and "exists" in out["error"]
    code, out = run_cli(["model", "new", "x", "--from", "nobody", "--dir", str(tmp_path / "x")], capsys)
    assert code == 2 and "bases: girl" in out["error"] and not (tmp_path / "x").exists()


def have_blender():
    try:
        cfg = CFG.load()
    except Exception:  # noqa: BLE001
        return False
    return Path(cfg["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")


@pytest.mark.skipif(not have_blender(), reason="Blender with mmd_tools not available")
def test_mannequin_end_to_end_with_blender(tmp_path, capsys):
    spec = mannequin_spec(tmp_path)
    code, out = run_cli(["model", "build", str(spec)], capsys)
    assert code == 0, out.get("error") or out.get("verify", {}).get("problems")
    assert out["ok"] is True and out["verify"]["ok"] is True and out["verify"]["problems"] == []
    v = out["verify"]["numbers"]
    assert v["pos_err_mm"] < 0.1 and v["bone_head_err_mm"] < 0.1 and v["morph_err_mm"] < 0.1
    assert v["weight_bad_vertices"] == 0 and v["faces_flipped"] == 0 and v["material_faces_ok"]
    assert v["bodies"] == v["bodies_expected"] == 26 and v["joints"] == v["joints_expected"] == 7
    rig = out["rig"]
    assert rig["missing_required"] == [] and rig["kind"] == "character"
    assert rig["morph_map"]["blink"] == "まばたき" and rig["morph_map"]["a"] == "あ"
    assert rig["chain_families"]["bangs"] == {"chains": 1, "bones": 3}
    assert rig["chain_families"]["skirt"] == {"chains": 2, "bones": 4}
    assert rig["bodies"] == 26 and rig["dynamic_bodies"] == 7
    d = Path(out["out"])
    for f in ("mq.pmx", "mq.blend", "mq.rig.json", "build.json", "tex/mannequin_skin.png"):
        assert (d / f).exists(), f
    rj = json.loads((d / "mq.rig.json").read_text(encoding="utf-8"))
    assert rj["schema"] == 1 and rj["map"]["head"] and rj["morphs"]["blink"] == "まばたき"
