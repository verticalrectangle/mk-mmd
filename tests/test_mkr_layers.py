"""mk render's layer request for mkr (mkmmd.render.mkr): patterns resolved on the bake's records, the keep table's PMX
names found on the cast member's meshes, the split of one output's layers between mkr and Blender (a frame with
negative screen type done with both its layers, as the plan lists them and a render promises them), and the looks: the
objects sorted as styles.Looks sorts them, the look jobs of frames and layers, and their passes composed."""
import json

import numpy as np
import pytest

from mkmmd.render import mkr as MKR


def _doc():
    mats = {"eye": {"settings": {"mmd_material": {"name_j": "瞳"}}},
            "eye.001": {"settings": {"mmd_material": {"name_j": "瞳"}}},
            "skin": {"settings": {"mmd_material": {"name_j": "肌"}}},
            "prop": {"settings": {}}}
    return {"objects": [{"name": "Reisen_mesh0", "collections": ["Reisen"], "props": []},
                        {"name": "swarm_swarm", "collections": ["Swarm"], "props": ["mk_swarm"]},
                        {"name": "guitar_body", "collections": ["Props"], "props": []}],
            "lit": {"records": {"lamp": {"collections": ["Lights"], "props": []}},
                    "materials": mats,
                    "meshes": [{"name": "Reisen_mesh0", "armature": "Reisen_arm", "materials": ["skin", "eye", None]},
                               {"name": "Reisen_guitar_mesh0", "armature": "Reisen_guitar_arm", "materials": ["eye.001"]},
                               {"name": "guitar_body", "armature": None, "materials": ["prop"]}],
                    "screen": {"meshes": []}}}


def test_figure_and_bare_patterns_resolve_on_the_bake():
    dem = {120: [{"kind": "figure", "key": "f1", "objects": ["Reisen_mesh*", "@Swarm"], "shot": "s"},
                 {"kind": "bare", "shot": "s", "key": "g0", "hide": ["prop:mk_swarm", "lamp"]},
                 {"kind": "point", "key": "x1", "expr": "1", "shot": "s"}]}
    req = MKR.request(_doc(), dem, None, [120, 121])
    items = req["items"]["120"]
    assert items[0]["names"] == ["Reisen_mesh0", "swarm_swarm"]
    assert items[1]["names"] == ["swarm_swarm", "lamp"]
    assert "names" not in items[2]
    assert req["keep"] is None
    assert req["screen"] == [120, 121]                    # the bake has a screen layer: every frame is asked


def test_keep_names_the_members_own_materials():
    keep = {"armature": "Reisen_arm", "materials": ["瞳"]}
    assert MKR.keep_materials(_doc(), keep) == ["eye"]   # not the guitarist's copy of the eye
    with pytest.raises(ValueError, match="has no material 髪"):
        MKR.keep_materials(_doc(), {"armature": "Reisen_arm", "materials": ["髪"]})


def test_split_gives_blender_what_mkr_cannot_draw_and_skips_what_is_done(tmp_path):
    (tmp_path / "point" / "x1").mkdir(parents=True)
    (tmp_path / "point" / "x1" / "00120.json").write_text('{"p": [0.5, 0.5], "depth": 1.0, "m": 0.1}')
    req = {"items": {"120": [{"kind": "point", "key": "x1", "expr": "1", "shot": "s"},
                             {"kind": "figure", "key": "f1", "objects": ["a"], "shot": "s", "names": ["a"]},
                             {"kind": "matte", "shot": "sil"}]},
           "keep": {"materials": ["eye"], "frames": [120, 121]}, "screen": [120, 121]}
    plan = {"layers": {120: [None, None, "a matte is drawn in its shot's flat look"]},
            "keep": {"draw": [120], "blender": {121: "why"}},
            "screen": {"draw": [121], "blender": {120: "screen type over the cap"}}}
    mine, theirs = MKR.split(req, plan, tmp_path)
    assert mine["items"] == {"120": [req["items"]["120"][1]]}          # the point is on disk already
    assert theirs["demands"] == {120: [{"kind": "matte", "shot": "sil"}]}
    assert mine["keep"] == {"materials": ["eye"], "frames": [120]} and mine["screen"] == [121]
    assert theirs["mk_keep"] == [120] and theirs["keep"] == [121]
    assert theirs["mk_screen"] == [121] and theirs["screen"] == [120]


def test_a_frame_with_negative_type_is_done_with_both_its_layers(tmp_path):
    for rel in ("screen/00121.png", "screen/00122.png", "negative/00122.png", "screen/00123.png"):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_bytes(b"png")
    req = {"items": {}, "keep": None, "screen": [120, 121, 122, 123, 124]}
    plan = {"screen": {"draw": [120, 121, 122, 123], "blender": {124: "why"}, "negative": [121, 122]}}
    mine, theirs = MKR.split(req, plan, tmp_path)
    assert mine["screen"] == [120, 121]                      # 121: its screen layer is there, its negative one is not
    assert theirs["mk_screen"] == [120, 121, 122, 123] and theirs["screen"] == [124]
    # a plan without negative frames (an mkr from before them): the screen layer is all a frame has
    mine, _ = MKR.split(req, {"screen": {"draw": [121, 122], "blender": {}}}, tmp_path)
    assert mine["screen"] is None


def test_the_plan_lists_mkrs_frames_with_negative_type(monkeypatch):
    def output(screen):
        return {"draw": [120, 121], "blender": {}, "layers": {}, "keep": {"draw": [], "blender": {}}, "screen": screen}
    doc = {"outputs": {"16x9": output({"draw": ["120", 121], "blender": {}, "negative": [121]}),
                       "9x16": output({"draw": [120], "blender": {"121": "why"}})},     # no "negative": none
           "unsupported": {}}
    monkeypatch.setattr(MKR, "_call", lambda argv, timeout, layers=None: doc)
    pl = MKR.plan("bake", ["16x9", "9x16"], [120, 121], {"items": {}, "keep": None, "screen": [120, 121]})
    assert pl["outputs"]["16x9"]["screen"] == {"draw": [120, 121], "blender": {}, "negative": [121]}
    assert pl["outputs"]["9x16"]["screen"] == {"draw": [120], "blender": {121: "why"}, "negative": []}
    assert "negative" not in pl["outputs"]["16x9"]["keep"]


def test_a_render_promises_the_negative_layer_of_mkrs_frames_with_negative_type():
    from mkmmd.cli.render import promise
    req = {"items": {}, "keep": None, "screen": [120, 121, 122]}
    out_plan = {"draw": [120, 121, 122], "screen": {"draw": [120, 121], "blender": {122: "why"}, "negative": [121]}}
    got = promise(req, out_plan, [120, 121, 122, 123])
    assert got[121] == ["00121.png", "negative/00121.png", "screen/00121.png"]
    assert got[120] == ["00120.png", "screen/00120.png"] and got[122] == ["00122.png", "screen/00122.png"]
    assert got[123] == ["00123.png"]


def test_frame_specs_join_runs():
    assert MKR.spec([5, 3, 4, 9, 11, 10, 20]) == "3:5,9:11,20"
    assert MKR.empty({"items": {}, "keep": None, "screen": None})
    assert not MKR.empty({"items": {}, "keep": None, "screen": [3]})


# ================================================================================================= the looks
def _look_doc():
    sil = {"colors": {"background": [0.3, 0.5, 0.6], "subject": [0.12, 0.11, 0.18], "accent": [0.98, 0.96, 0.93]},
           "hide": ["*"], "keep": ["Reisen*", "cord"], "accent": ["cord", "bolt"], "tint": [], "knockout": None,
           "samples": 16, "grow": 1.0}
    objs = [{"name": "Reisen_mesh0", "collections": ["Cast"], "props": [], "text": False, "soft": False,
             "materials": ["skin", "eye"]},
            {"name": "cord", "collections": [], "props": [], "text": False, "soft": False, "materials": ["cord"]},
            {"name": "bolt", "collections": [], "props": [], "text": False, "soft": True, "materials": ["bolt"]},
            {"name": "wall", "collections": ["Sets"], "props": [], "text": False, "soft": False, "materials": ["wall"]}]
    return {"frame0": 100, "fps": 30.0, "objects": objs, "outputs": [{"name": "16x9", "size": [1920, 1080]}],
            "shots": [{"name": "lit", "from": 100, "to": 110, "cameras": {"16x9": "lit@16x9"}},
                      {"name": "sil", "from": 110, "to": 120, "cameras": {"16x9": "sil@16x9"},
                       "styles": {"16x9": {"silhouette": sil}}}],
            "medium": [], "looks": {"tints": []}, "lit": {"meshes": [], "records": {}, "materials": {}}}


def test_a_silhouette_sorts_its_objects_as_blender_does():
    doc = _look_doc()
    name, kind, spec = MKR.look_at(doc, "16x9", 112)
    assert (name, kind) == ("sil", "silhouette")
    assert MKR.look_at(doc, "16x9", 105)[1] is None
    cl = MKR.look_sort(doc, kind, spec)
    assert cl["subject"] == ["Reisen_mesh0"] and cl["hard"] == ["cord"] and cl["hide"] == ["bolt", "wall"]
    # a glitch's hidden objects leave every class for `hide`
    assert MKR.look_sort(doc, kind, spec, hidden=["cord"])["hard"] == []


def test_look_jobs_cover_the_frames_and_the_layers_seen_through_a_look(tmp_path):
    doc = _look_doc()
    items = {"108": [{"kind": "matte", "shot": "sil"}, {"kind": "plate", "shot": "lit"}],
             "115": [{"kind": "figure", "key": "f", "objects": ["*"], "shot": "sil", "names": ["Reisen_mesh0", "wall"]}]}
    lk = MKR.look_jobs(doc, "16x9", [105, 112], items)
    assert list(lk["frames"]) == ["112"] and lk["jobs"][0]["role"] == "frame"
    assert items["108"][0]["looks"] == {"16x9": 1} and "looks" not in items["108"][1]
    assert items["115"][0]["look_names"] == {"16x9": ["Reisen_mesh0"]}         # the look hides the wall
    st = lk["sets"][0]
    assert st["kind"] == "silhouette" and st["hard"] == ["cord"]
    assert abs(st["subject_color"][0] - ((0.12 + 0.055) / 1.055) ** 2.4) < 1e-6    # linear, as the Object colour is set
    req = {"items": items, "keep": None, "screen": None, "looks": {"outputs": {"16x9": lk}}}
    mine, theirs = MKR.split(req, {"draw": [105, 112], "layers": {108: [None, None], 115: [None]}}, tmp_path, "16x9")
    assert MKR.look_ids(mine, "16x9") == [0, 1]
    jobs = MKR.compose_jobs(doc, np.zeros(16, np.float32), "16x9", lk, [1920, 1080], 100, tmp_path, [0, 1])
    assert jobs[0]["files"] == [str(tmp_path / "00112.png")] and jobs[0]["hard"] is True
    assert jobs[1]["files"] == [str(tmp_path / "back" / "sil" / "00108.png"), str(tmp_path / "matte" / "sil" / "00108.png")]
    assert jobs[1]["spec"]["knockout"]["color"] == doc["shots"][1]["styles"]["16x9"]["silhouette"]["colors"]["subject"]


def test_a_matte_job_writes_the_back_then_the_coverage(tmp_path):
    from mkmmd.render import compose as CO
    h, w = 4, 6
    flat = np.zeros((h, w, 4), np.uint8)
    flat[1:3, 2:4] = [32, 30, 47, 254]
    matte = np.zeros((2 * h, 2 * w), np.uint8)
    matte[2:6, 4:8] = 254
    head = json.dumps({"job": 3, "passes": {"flat": {"offset": 256, "shape": [h, w, 4], "type": "u8"},
                                            "matte": {"offset": 512, "shape": [2 * h, 2 * w], "type": "u8"}}}).encode()
    buf = bytearray(b"MKRLOOK1" + len(head).to_bytes(4, "little") + head)
    buf += bytes(256 - len(buf)) + flat.tobytes()
    buf += bytes(512 - len(buf)) + matte.tobytes()
    (tmp_path / "3.look").write_bytes(bytes(buf))
    spec = _look_doc()["shots"][1]["styles"]["16x9"]["silhouette"]
    job = {"id": 3, "look": "silhouette", "role": "matte", "spec": spec, "tints": [], "hard": False,
           "files": [str(tmp_path / "back.png"), str(tmp_path / "matte.png")]}
    CO.finish_look(str(tmp_path / "3.look"), job)
    assert not (tmp_path / "3.look").exists()
    from PIL import Image
    back = np.asarray(Image.open(tmp_path / "back.png"))
    assert (back == np.floor(np.array(spec["colors"]["background"]) * 255 + 0.5)).all()     # no figure in the back
    m = np.asarray(Image.open(tmp_path / "matte.png"))
    assert m[3, 5, 0] == 255 and m[0, 0, 0] == 0                                        # 254 is a whole pixel
