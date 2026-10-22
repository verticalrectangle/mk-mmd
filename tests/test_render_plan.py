"""mk's own renderer, the plan (mkmmd.render.plan), bpy-free: which shots it draws from a bake and which it leaves to
Blender (and why), the vector look's materials numbered as Blender numbers them, and a ring's centre seen through the
bake's camera as Blender's world_to_camera_view sees it. The drawing itself is checked against Blender's frames in
test_build_stages.py."""
import json

import numpy as np
import pytest

from mkmmd.core import palette as PAL
from mkmmd.core import shotstyle as SS
from mkmmd.render import plan as PL

DAWN = PAL.get("rose-pine-dawn")
VECTOR = {"tones": {"fill": {"lit": "text", "shade": "subtle"}, "eye": {"colors": ["text", "base"], "at": [0.5]}},
          "materials": [{"match": "eye*", "tone": "eye"}]}
FRAMES = 10


def styles(**shot):
    return SS.normalize(shot, DAWN, VECTOR)


def obj(name, mats, baked="glb", shows=((0, FRAMES),), text=False, why=None):
    return {"name": name, "type": "MESH", "collections": [], "props": [], "text": text, "materials": list(mats),
            "shadow": True, "baked": baked, **({"why": why} if why else {}), "shows": [list(r) for r in shows]}


def shot(name, start, look):
    return {"name": name, "from": start, "to": start + 5, "styles": {"o": look} if look else {}, "cameras": {"o": "cam"}}


def bake(objects, shots, materials=None, blocks=(), rings=None, errors=None, shift=(0.0, 0.0), point=None):
    """A bake of FRAMES frames from frame 1 at 10 fps: one camera, Blender's identity (it looks down -Z, glTF's -Y),
    a 36 mm lens on its 36 mm sensor, its lens `shift`, clipped from 0.1 to 100 m; a ring's point at glTF `point` on
    every frame."""
    cam = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 36.0, *shift], np.float32)
    blob, points = [np.tile(cam, FRAMES), np.tile(np.float32([0.1, 100.0]), FRAMES)], {}
    if point is not None:
        points["0"] = 17 * FRAMES
        blob.append(np.tile(np.asarray(point, np.float32), FRAMES))
    doc = {"version": PL.BAKE_VERSION, "fps": 10, "frame0": 1, "frames": FRAMES, "shots": shots, "objects": objects,
           "cameras": [{"name": "cam", "sensor": 36.0, "sensor_height": 24.0, "fit": "AUTO", "offset": 0,
                        "clip_at": 15 * FRAMES}],
           "blocks": list(blocks), "materials": materials if materials is not None else {"eye_l": {"texture": "glb"}},
           "rings": {"table": rings, "points": points, "errors": errors or {}}}
    return doc, np.concatenate(blob)


def plan(doc, blob):
    return PL.plan(doc, blob, [("o", [100, 50])], list(range(1, FRAMES + 1)), 50)


def test_a_vector_shot_with_all_it_shows_baked_is_drawn_by_mk_with_its_materials_numbered_as_blender_numbers_them():
    look = styles(style="vector")
    doc, blob = bake([obj("body", ["skin", "cloth"]), obj("eyes", ["eye_l", "skin"])], [shot("a", 1, look)])
    p = plan(doc, blob)
    assert [it["frame"] for it in p["items"]] == list(range(1, 11)) and p["blender"] == {"o": []}
    assert p["shots"] == {"o": {"a": {"engine": "mk", "frames": 10}}}
    lk = p["looks"][0]
    assert lk["ids"] == {"skin": 1, "cloth": 2, "eye_l": 3}
    assert lk["table"] == [list(t) for t in SS.vector_table(["skin", "cloth", "eye_l"], look["vector"])]
    assert lk["table"][0] == [None, "fill", "fill", "eye"] and lk["tex"] is True       # the eye tone is drawn
    assert sorted(lk["draw"]) == ["body", "eyes"]
    assert p["items"][0]["size"] == [100, 50]                    # 50 x 25 at 50 %, the passes at twice that


CORD = obj("cord", ["rope"], baked=None, why="a curves object", shows=[(3, 5)])
WORDS = obj("words", ["ink"], baked=None, text=True)
FAILING = {"fps": 10, "frame0": 1, "rings": [{"index": 0, "at": 0.2, "until": 0.6, "center": {"mode": "point", "expr": "x"}}]}


@pytest.mark.parametrize("change, why", [
    ({"objects": [CORD]}, "cord: a curves object"),
    ({"objects": [dict(CORD, shows=[])]}, None),                                   # it never shows in the shot
    ({"objects": [WORDS]}, "words: type over the picture"),
    ({"objects": [WORDS], "look": styles(style="vector", hide=["words"])}, None),
    ({"look": styles(style="silhouette")}, "a silhouette look"),
    ({"look": {}}, "lit by the scene"),
    ({"materials": {"eye_l": {"texture": "other"}}}, "material eye_l: drawn from a texture the bake does not hold"),
    ({"objects": [obj("eyes", ["eye_l"], baked="by_frame")]}, "material eye_l: drawn from a texture on a mesh baked"),
    ({"blocks": [{"name": "crowd", "why": "instances collection 'crowd'", "shows": [[0, 2]]}]}, "crowd: instances"),
    ({"rings": FAILING, "errors": {"0": "ring 0: center 'x' failed at frame 3: NameError"}}, "ring 0: center 'x' failed"),
])
def test_a_shot_mk_cannot_draw_goes_whole_to_blender_with_the_reason_in_the_report(change, why):
    objects = {o["name"]: o for o in [obj("body", ["skin"]), obj("eyes", ["eye_l"])]}
    objects.update({o["name"]: o for o in change.get("objects", [])})
    doc, blob = bake(list(objects.values()), [shot("a", 1, change.get("look", styles(style="vector")))],
                     materials=change.get("materials"), blocks=change.get("blocks", ()), rings=change.get("rings"),
                     errors=change.get("errors"))
    p = plan(doc, blob)
    rec = p["shots"]["o"]["a"]
    if why is None:
        assert rec["engine"] == "mk" and len(p["items"]) == FRAMES
    else:
        assert rec["engine"] == "blender" and why in rec["why"]
        assert p["items"] == [] and p["blender"]["o"] == list(range(1, 11))


def test_each_shot_is_decided_on_its_own():
    doc, blob = bake([obj("body", ["skin"])], [shot("a", 1, styles(style="vector")), shot("b", 6, styles(style="silhouette"))])
    p = plan(doc, blob)
    assert [it["frame"] for it in p["items"]] == [1, 2, 3, 4, 5] and p["blender"]["o"] == [6, 7, 8, 9, 10]
    assert p["shots"]["o"]["b"] == {"engine": "blender", "frames": 5, "why": "a silhouette look"}


def test_the_frames_of_a_medium_window_with_a_look_go_to_blender_and_the_rest_of_the_shot_stays_mk_s():
    doc, blob = bake([obj("body", ["skin"])], [shot("a", 1, styles(style="vector"))])
    doc["medium"] = [{"index": 0, "from": 4, "to": 7, "styles": {"o": styles(style="vector", sketch=True)}},
                     {"index": 1, "from": 8, "to": 10, "styles": {"o": {}}}]           # this output keeps the shot's
    p = plan(doc, blob)
    assert p["blender"]["o"] == [4, 5, 6] and [it["frame"] for it in p["items"]] == [1, 2, 3, 7, 8, 9, 10]
    assert p["shots"]["o"]["medium 0"] == {"engine": "blender", "frames": 3, "why": "a [[medium]] window's look"}


def test_a_cel_tone_asks_for_the_texture_pass_and_a_flat_one_does_not():
    cel = SS.normalize({"style": "vector", "materials": [{"match": "skin", "tone": "cel"}],
                        "tones": {"cel": {"texture": True, "shade": "subtle"}}}, DAWN, VECTOR)
    for look, tex in ((cel, True), (styles(style="vector"), False)):
        doc, blob = bake([obj("body", ["skin"])], [shot("a", 1, look)], materials={})
        assert plan(doc, blob)["looks"][0]["tex"] is tex



@pytest.mark.parametrize("shift, point, seen", [
    ((0.0, 0.0), (1.0, -5.0, 0.0), (0.7, 0.5)),   # 1 m right at 5 m, where half the frame is 2.5 m wide
    ((0.1, 0.0), (1.0, -5.0, 0.0), (0.6, 0.5)),   # the lens shifted right a tenth of the frame: further left
    ((0.0, 0.1), (1.0, -5.0, 0.0), (0.7, 0.6)),   # shifted up: lower
    ((0.0, 0.0), (0.0, -5.0, -2.0), (0.5, 0.1)),  # glTF's -Z is Blender's +Y, this camera's up: near the top
    ((0.0, 0.0), (0.0, 5.0, 0.0), None),          # behind the camera
])
def test_a_ring_s_centre_is_where_blender_s_camera_sees_it(shift, point, seen):
    rings = {"fps": 10, "frame0": 1, "rings": [{"index": 0, "at": 0.0, "until": 9.0, "center": {"mode": "point", "expr": "p"}}]}
    doc, blob = bake([obj("body", ["skin"])], [shot("a", 1, styles(style="vector"))], rings=rings, shift=shift, point=point)
    c = PL.plan(doc, blob, [("o", [100, 100])], [3], 100)["items"][0]["centres"][0]
    assert c is None if seen is None else c == pytest.approx(seen, abs=1e-6)


def test_an_older_bake_is_refused_with_the_command_that_bakes_it_again(tmp_path):
    (tmp_path / "bake.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
    (tmp_path / "bake.bin").write_bytes(b"")
    with pytest.raises(PL.BakeError, match="mk build --bake"):
        PL.load(tmp_path)
