"""Medium shifts (mkmmd.core.medium): [[medium]] windows normalised per output like a shot's look, in time order, one at
a time and inside the clip. Blender drawing a window's look over the shot in the cut is checked in test_build_stages.py."""
import pytest

from mkmmd.core import medium as MD
from mkmmd.core import palette as PAL
from mkmmd.core import shotstyle as SS

MOON = PAL.get("rose-pine-moon")
OUTS = ["16x9", "9x16"]


def norm(entries, last=None):
    return MD.normalize(entries, OUTS, MOON, None, 30, 1, last)


def test_a_window_is_a_look_per_output_held_over_its_frames_and_an_aspect_can_leave_an_output_to_the_shot():
    ws = norm([{"from": 2.0, "to": 2.5, "style": "vector", "sketch": True, "aspect": {"9x16": {"style": "none"}}},
               {"from": 0.5, "to": 1.0, "style": "silhouette", "colors": {"subject": "love"},
                "aspect": {"9x16": {"colors": {"background": "gold"}}}}])
    assert [(w["index"], w["from"], w["to"]) for w in ws] == [(1, 16, 31), (0, 61, 76)]      # time order, frames
    sil = ws[0]["styles"]["9x16"]["silhouette"]["colors"]
    assert sil["subject"] == pytest.approx(SS.resolve_colour("love", MOON), abs=1e-6)         # the window's own ...
    assert sil["background"] == pytest.approx(SS.resolve_colour("gold", MOON), abs=1e-6)      # ... the aspect's laid on
    assert ws[1]["styles"]["16x9"]["vector"]["sketch"]["fps"] == 12.0 and ws[1]["styles"]["9x16"] == {}
    assert [MD.at(ws, f) and MD.at(ws, f)["index"] for f in (15, 16, 30, 31, 61, 75, 76)] == [None, 1, 1, None, 0, 0, None]


@pytest.mark.parametrize("entries, frag", [
    ([{"from": 0.0, "to": 1.0, "style": "vector"}, {"from": 0.5, "to": 1.5, "style": "silhouette"}], "overlap"),
    ([{"from": 0.0, "to": 1.0}], "a window is drawn in a look"),
    ([{"from": 0.0, "to": 1.0, "style": "vector", "reflection": {"object": "glass"}}], "unknown keys ['reflection']"),
    ([{"from": 0.0, "to": 1.0, "style": "vector", "aspect": {"4x3": {}}}], "an output the project does not have"),
    ([{"from": 0.0, "to": 1.0, "style": "vector", "aspect": {"9x16": {"lens": 30}}}], "unknown keys ['lens']"),
    ([{"from": 0.0, "to": 1.0, "style": "silhouette", "sketch": True}], "medium 0 (16x9): sketch belong to the vector"),
    ([{"from": 1.0, "to": 1.01, "style": "vector"}], "under a frame"),
    ([{"from": 9.0, "to": 11.0, "style": "vector"}], "outside the clip"),
    ([{"from": "a", "to": 1.0, "style": "vector"}], "clip seconds")])
def test_bad_windows_say_what_is_wrong(entries, frag):
    with pytest.raises(MD.MediumError, match=frag.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(")
                       .replace(")", r"\)")):
        norm(entries, last=300)
