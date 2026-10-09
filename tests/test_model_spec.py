"""mkmmd.model.spec: TOML loading with include lists, merging, overrides."""
from pathlib import Path

import pytest

from mkmmd.model import spec as SP


def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_includes_merge_in_order_and_main_wins(tmp_path):
    write(tmp_path / "a.toml", '[hair]\nlength = 0.3\ncolor = "red"\n[body]\nheight = 1.6\n')
    write(tmp_path / "b.toml", '[hair]\nlength = 0.4\nparts = [1, 2]\n')
    main = write(tmp_path / "model.toml",
                 '[model]\nname = "x"\nparts = ["body"]\ninclude = ["a.toml", "b.toml"]\n[hair]\ncolor = "blue"\n')
    s = SP.load(main)
    assert s["hair"] == {"length": 0.4, "color": "blue", "parts": [1, 2]}      # b over a, main over both
    assert s["body"]["height"] == 1.6
    assert "include" not in s["model"]
    assert [f.name for f in s.files] == ["a.toml", "b.toml", "model.toml"]
    assert s.path == main.resolve() and s.dir == tmp_path.resolve()


def test_lists_replace_and_tables_merge():
    a = {"t": {"x": 1, "y": {"p": 1, "q": 2}}, "l": [1, 2, 3]}
    b = {"t": {"y": {"q": 9}, "z": 3}, "l": [7]}
    m = SP.merge(a, b)
    assert m == {"t": {"x": 1, "y": {"p": 1, "q": 9}, "z": 3}, "l": [7]}
    assert a["t"]["y"]["q"] == 2                                              # inputs untouched


def test_nested_includes_resolve_relative_to_the_including_file(tmp_path):
    write(tmp_path / "sub" / "deep.toml", "[a]\nv = 1\n")
    write(tmp_path / "sub" / "mid.toml", '[model]\ninclude = ["deep.toml"]\n[a]\nw = 2\n')
    main = write(tmp_path / "m.toml", '[model]\nparts = ["p"]\ninclude = ["sub/mid.toml"]\n')
    s = SP.load(main)
    assert s["a"] == {"v": 1, "w": 2}


def test_include_cycle_and_missing_file(tmp_path):
    write(tmp_path / "a.toml", '[model]\ninclude = ["b.toml"]\n')
    write(tmp_path / "b.toml", '[model]\ninclude = ["a.toml"]\n')
    with pytest.raises(SP.SpecError, match="cycle"):
        SP.load(tmp_path / "a.toml")
    write(tmp_path / "c.toml", '[model]\ninclude = ["nope.toml"]\n')
    with pytest.raises(SP.SpecError, match="nope.toml"):
        SP.load(tmp_path / "c.toml")
    write(tmp_path / "bad.toml", "[model\n")
    with pytest.raises(SP.SpecError):
        SP.load(tmp_path / "bad.toml")


def test_tilde_and_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write(tmp_path / "shared" / "colors.toml", '[colors]\nhair = "#d8262b"\n')
    main = write(tmp_path / "proj" / "model.toml",
                 '[model]\nname = "rin"\nparts = ["body"]\ninclude = ["~/shared/colors.toml"]\nout = "~/models/rin"\n'
                 '[misc]\nfolder = "rel/dir"\n')
    s = SP.load(main)
    assert s["colors"]["hair"] == "#d8262b"
    cfg = SP.model_cfg(s)
    assert cfg["out"] == tmp_path / "models" / "rin"
    assert s.get_path("misc.folder") == (tmp_path / "proj").resolve() / "rel" / "dir"


def test_overrides_and_values(tmp_path):
    main = write(tmp_path / "m.toml", '[model]\nparts = ["p"]\n[hair]\nlength = 0.3\n')
    s = SP.load(main, ["hair.length=0.5", "hair.tie=true", 'hair.name="x y"', "new.deep.key=[1, 2]", "hair.raw=plain"])
    assert s["hair"] == {"length": 0.5, "tie": True, "name": "x y", "raw": "plain"}
    assert s["new"]["deep"]["key"] == [1, 2]
    with pytest.raises(SP.SpecError):
        SP.apply_overrides(s, ["novalue"])


def test_model_cfg_defaults_and_errors():
    s = SP.from_dict({"model": {"name": "n", "parts": ["a", "b"]}})
    c = SP.model_cfg(s)
    assert c["parts"] == ["a", "b"] and c["seed"] == 1 and c["scale"] == 0.08
    assert c["out"].name == "n" and c["builders"] == {} and c["needs"] == {}
    with pytest.raises(SP.SpecError):
        SP.model_cfg(SP.from_dict({"model": {"name": "n"}}))


def test_digest_is_stable_and_content_based():
    a = SP.from_dict({"b": 1, "a": {"y": 2, "x": 1}})
    b = SP.from_dict({"a": {"x": 1, "y": 2}, "b": 1})
    c = SP.from_dict({"a": {"x": 1, "y": 3}, "b": 1})
    assert a.digest() == b.digest() != c.digest()
    assert SP.dig(a, "a.y") == 2 and SP.dig(a, "a.zzz", 5) == 5 and a.lookup("b") == 1


def test_bases_load_by_name_under_the_file(tmp_path):
    """`base:NAME` merges a model base shipped with mk under the including file (its own keys win, the base's other keys
    stay), names a file inside the base, and loads the base as a main spec; an unknown base is an error listing them."""
    main = write(tmp_path / "c" / "model.toml", '[model]\nname = "c"\ninclude = ["base:girl"]\n[hair.bangs]\ncount = 3\n')
    s, base = SP.load(main), SP.load("base:girl")
    assert s["model"]["name"] == "c" and s["model"]["parts"] == base["model"]["parts"]
    assert s["hair"]["bangs"]["count"] == 3 != base["hair"]["bangs"]["count"]
    assert s["hair"]["bangs"]["above_eye"] == base["hair"]["bangs"]["above_eye"] and s["colors"] == base["colors"]
    assert s.files == (*base.files, main.resolve())
    assert SP.expand("base:girl/hand.npz").is_file() and SP.expand("base:girl") == base.path
    with pytest.raises(SP.SpecError, match="no model base 'nobody'; bases: girl"):
        SP.load(write(tmp_path / "x.toml", '[model]\ninclude = ["base:nobody"]\n'))
