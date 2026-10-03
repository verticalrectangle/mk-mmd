import json

import numpy as np
import pytest

from mkmmd import cache
from mkmmd.core import bonemap, frames, jsonx
from mkmmd.project import Project, ProjectError


# ---------------------------------------------------------------- frames
def test_frames_ranges_lists_and_steps():
    assert frames.parse("10:13") == [10, 11, 12, 13]
    assert frames.parse("10:20:5") == [10, 15, 20]
    assert frames.parse("100,140,200") == [100, 140, 200]
    assert frames.parse("5,10:11") == [5, 10, 11]


def test_frames_clip_time_and_all_need_a_project():
    with pytest.raises(frames.FrameSpecError):
        frames.parse("all")
    assert frames.parse("t=0:0.1", fps=30, frame0=181) == [181, 182, 183, 184]
    assert frames.parse("t=1.0", fps=30, frame0=181) == [211]
    assert frames.parse("all", fps=30, frame0=181, duration=0.1) == [181, 182, 183]


def test_frames_rejects_backwards_ranges():
    with pytest.raises(frames.FrameSpecError):
        frames.parse("20:10")


# ---------------------------------------------------------------- bone map
def test_bonemap_matches_full_width_and_sides():
    bones = {"腕.L": "左腕", "人指１.R": "右人指１", "足ＩＫ.L": "左足ＩＫ", "頭": "頭", "上半身2": "上半身2",
             "腰キャンセル.L": "腰キャンセル左"}
    m = bonemap.build_map(bones)
    assert m["arm.L"] == "腕.L"
    assert m["index1.R"] == "人指１.R"
    assert m["leg_ik.L"] == "足ＩＫ.L"
    assert m["head"] == "頭"
    assert m["upper_body2"] == "上半身2"
    assert m["waist_cancel.L"] == "腰キャンセル.L"
    assert "arm.R" not in m


def test_bonemap_falls_back_to_blender_names_without_pmx_names():
    m = bonemap.build_map({"手首.R": "", "ひじ.R": ""})
    assert m["wrist.R"] == "手首.R" and m["elbow.R"] == "ひじ.R"


def test_bonemap_does_not_confuse_leg_ankle_and_leg_ik():
    m = bonemap.build_map({"足.L": "左足", "足首.L": "左足首", "足ＩＫ.L": "左足ＩＫ"})
    assert (m["leg.L"], m["ankle.L"], m["leg_ik.L"]) == ("足.L", "足首.L", "足ＩＫ.L")


# ---------------------------------------------------------------- project
def test_project_loads_and_maps_time(tmp_path):
    (tmp_path / "mk.toml").write_text(
        '[project]\nname = "demo"\nfps = 30\nframe0 = 181\nduration = 24.6\nblend = "build/x.blend"\n'
        '[[output]]\nname = "9x16"\nsize = [1080, 1920]\n', encoding="utf-8")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    p = Project.find(sub)
    assert p.name == "demo" and p.n_frames == 738 and p.last_frame == 918
    assert p.frame(1.0) == 211 and p.time(211) == 1.0
    assert p.blend == tmp_path / "build" / "x.blend"
    assert p.output("9x16").size == (1080, 1920)
    with pytest.raises(ProjectError):
        p.output("16x9")


def test_project_requires_time_fields(tmp_path):
    (tmp_path / "mk.toml").write_text('[project]\nname = "x"\nfps = 30\n', encoding="utf-8")
    with pytest.raises(ProjectError):
        Project.load(tmp_path)


# ---------------------------------------------------------------- cache + json
def test_cache_key_tracks_file_content(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one")
    k1 = cache.key("op", {"x": 1}, f)
    f.write_text("two")
    assert cache.key("op", {"x": 1}, f) != k1
    assert cache.key("op", {"x": 1}, f) == cache.key("op", {"x": 1}, f)


def test_cache_roundtrip(tmp_path):
    c = cache.Cache(tmp_path)
    c.save_npz("hair", "k", {"x": np.arange(3.0)})
    assert c.load_npz("hair", "k")["x"].tolist() == [0.0, 1.0, 2.0]
    c.save_json("grip", "k", {"a": 1})
    assert c.load_json("grip", "k") == {"a": 1}
    assert c.load_json("grip", "missing") is None


def test_jsonx_converts_numpy_and_rounds():
    s = jsonx.dumps({"v": np.array([1.23456789, 2.0]), "n": np.float32(0.5), "bad": float("nan")}, precision=3)
    assert json.loads(s) == {"v": [1.235, 2.0], "n": 0.5, "bad": None}
