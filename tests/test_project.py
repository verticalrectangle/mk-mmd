"""The project file (docs/design.md: Project file, Includes and looks): files and looks that lie under the project's own
tables, a cut-out's grade from its clip's registry entry, and the paper-dawn look as the library reads it."""
import json

import numpy as np
import pytest

from mkmmd import post as POST
from mkmmd.core import palette as PAL
from mkmmd.core import shotstyle as SS
from mkmmd.project import Project, ProjectError

HEAD = "[project]\nfps = 30\nframe0 = 1\nduration = 2.0\n"


def project(tmp_path, body, **files):
    """The tables of a project whose mk.toml is HEAD and `body` (which goes on in `[project]`), beside `files`."""
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / "mk.toml").write_text(HEAD + body, encoding="utf-8")
    return Project.load(tmp_path).data


def test_an_included_file_lies_under_the_project_s_own_tables(tmp_path):
    base = ('[post]\nvignette = 0.0\ngrain = { amount = 0.0, size = 2.0 }\n[vector]\nlight = [1.0, 0.0, 0.0]\n'
            '[[vector.materials]]\nmatch = "*"\ntone = "skin"\n')
    data = project(tmp_path, 'include = "base.toml"\n[post]\ngrain = { size = 1.5 }\n[vector]\nlight = [0.0, 0.0, 1.0]\n'
                             '[[vector.materials]]\nmatch = "prop_*"\ntone = "set"\n', **{"base.toml": base})
    assert data["post"] == {"vignette": 0.0, "grain": {"amount": 0.0, "size": 1.5}}        # tables merge key by key
    assert data["vector"]["light"] == [0.0, 0.0, 1.0]                                       # a list of numbers is replaced
    assert [r["match"] for r in data["vector"]["materials"]] == ["prop_*", "*"]             # rules join, the project's first


def test_a_look_is_included_by_name_and_what_is_not_there_or_loops_is_refused(tmp_path):
    data = project(tmp_path, 'include = ["look:paper-dawn", "own.toml"]\n',
                   **{"own.toml": '[look]\nslots = { gold = "#f2c46d" }\n'})
    assert data["look"] == {"palette": "rose-pine-dawn", "slots": {"gold": "#f2c46d"}}
    with pytest.raises(ProjectError, match="no such look .*paper-dawn"):
        project(tmp_path, 'include = "look:paper-dusk"\n')
    with pytest.raises(ProjectError, match="missing.toml not found"):
        project(tmp_path, 'include = "missing.toml"\n')
    with pytest.raises(ProjectError, match="include loops"):
        project(tmp_path, 'include = "a.toml"\n', **{"a.toml": '[project]\ninclude = "mk.toml"\n'})


def test_the_paper_dawn_look_is_one_the_vector_look_takes_and_its_grade_changes_nothing(tmp_path):
    data = project(tmp_path, 'include = "look:paper-dawn"\n')
    palette = PAL.get(data["look"]["palette"])
    SS.normalize_vector({"style": "vector", "cutout": True}, palette, project=data["vector"])   # it refuses a bad tone
    frame = np.random.default_rng(1).random((36, 64, 3)).astype(np.float32)
    assert POST.Grade(data["post"], palette, (64, 36))(frame) == pytest.approx(frame, abs=1e-6)


def test_a_cutout_that_gives_no_grade_takes_the_one_stored_with_its_clip(tmp_path, monkeypatch):
    stored = {"gain": 1.4, "warm": 0.0, "saturation": 0.8, "lift": 0.2, "toward": "rose"}
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "registry.json").write_text(json.dumps([{"slug": "hand_a", "kind": "reference", "extra": {"grade": stored}},
                                                       {"slug": "hand_b", "kind": "reference"}]), encoding="utf-8")
    monkeypatch.setenv("MK_ASSETS", str(assets))
    cut = '[[fx]]\nkind = "cutout"\nfrom = 0.0\nto = 1.0\nclip = "{}"\n{}'
    hands = [("hand_a", ""), ("hand_a", "grade = { gain = 2.0 }\n"), ("hand_a", "grade = false\n"), ("hand_b", ""),
             ("footage/hand_a.mp4", "")]
    data = project(tmp_path, "".join(cut.format(*h) for h in hands))
    # the clip's; its own as written; as shot on its word; a clip with none stored; a file: the registry is not asked
    assert [fx.get("grade") for fx in data["fx"]] == [stored, {"gain": 2.0}, False, None, None]
