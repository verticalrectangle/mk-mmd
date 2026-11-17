"""[[look.span]] (mkmmd.core.lookspan): a part of the clip seen through another look holds the frames a [[shot]] with the
same numbers holds, one look a frame, inside the clip; a span that cannot be used says which and why. The build keying
the scene's view settings, the bake capturing each look and mkr drawing each frame through its own are checked on a
scratch project (docs/design.md: Lights and the look)."""
import pytest

from mkmmd.core import lookspan as LS
from mkmmd.core import palette as PAL
from mkmmd.core import shotstyle as SS
from mkmmd.core import transition as TR

FPS, FRAME0 = 30, 181
LAST = FRAME0 + 10 * FPS - 1                          # a 10 s clip


def norm(spans):
    return LS.normalize(spans, FPS, FRAME0, LAST)


def test_a_span_holds_the_frames_a_shot_with_the_same_numbers_holds():
    # seconds between frames: 7.656 s is frame 410.68, 8.69 s is 441.7; the cut there is on 411 and on 442
    shots = [{"name": "night", "from": 0.0, "to": 7.656}, {"name": "sweep", "from": 7.656, "to": 8.69},
             {"name": "end", "from": 8.69, "to": 10.0}]
    cuts = TR.plan({"shot": shots}, FPS, FRAME0, PAL.get("rose-pine-moon"))["cuts"]
    spans = norm([{"from": 7.656, "to": 8.69, "exposure": 0.5}, {"from": 0.0, "to": 7.656, "contrast": "High Contrast"}])
    assert [(s["index"], s["from"], s["to"]) for s in spans] == [(1, 181, 411), (0, 411, 442)]   # time order, frames
    assert spans[0]["contrast"] == "High Contrast" and spans[1]["exposure"] == 0.5 and "gamma" not in spans[1]
    for f in range(FRAME0, LAST + 1):
        held = [s["index"] for s in spans if s["from"] <= f < s["to"]]
        assert held == {"night": [1], "sweep": [0], "end": []}[SS.shot_at(cuts, f)["name"]], f
    assert norm([{"from": 8.69, "to": 10.0, "gamma": 1.2}])[0]["to"] - 1 == LAST     # up to the clip's last frame


def test_spans_side_by_side_are_one_look_a_frame():
    spans = norm([{"from": 0.0, "to": 2.0, "exposure": -1}, {"from": 2.0, "to": 3.0, "gamma": 1.1}])
    assert [(s["from"], s["to"]) for s in spans] == [(181, 241), (241, 271)]


@pytest.mark.parametrize("spans, message", [
    ([{"from": 0.0, "to": 1.0, "exposure": -1, "saturation": 2}],
     "look span 0: unknown keys ['saturation'] (known: from, to, exposure, contrast, gamma)"),
    ([{"from": 0.0, "to": 1.0, "exposure": -1}, {"from": 2.0, "to": 3.0, "view": "Standard"}],
     "look span 1: view: a project has one view transform"),
    ([{"from": 0.0, "to": 2.0, "exposure": -1}, {"from": 3.0, "to": 4.0, "gamma": 2},
      {"from": 1.5, "to": 2.5, "gamma": 1}], "look span 0 and look span 2 overlap in frames 226..240: one look a frame"),
    ([{"from": 9.5, "to": 10.2, "exposure": -1}], "look span 0 (9.5 .. 10.2 s): frames 466..486 reach outside the clip"),
    ([{"from": -0.5, "to": 1.0, "exposure": -1}], "look span 0 (-0.5 .. 1.0 s): frames 166..210 reach outside the clip"),
    ([{"from": 1.0, "to": 1.01, "exposure": -1}], "look span 0 (1.0 .. 1.01 s): under a frame"),
    ([{"from": 2.0, "to": 1.0, "exposure": -1}], "look span 0 (2.0 .. 1.0 s): from is not before to"),
    ([{"from": 0.0, "to": 1.0}], "look span 0 (0.0 .. 1.0 s): sets none of exposure, contrast, gamma"),
    ([{"from": "0", "to": 1.0, "exposure": -1}], "look span 0: from = '0': clip seconds"),
    ([{"from": 0.0, "to": True, "exposure": -1}], "look span 0: to = True: clip seconds"),
    ([{"from": 0.0, "to": 1.0, "exposure": "dark"}], "look span 0 (0.0 .. 1.0 s): exposure = 'dark': a number"),
    ([{"from": 0.0, "to": 1.0, "contrast": 3}], "look span 0 (0.0 .. 1.0 s): contrast = 3: the name of a Blender look"),
    ("dark", "[[look.span]]: expected an array of tables")])
def test_a_span_that_cannot_be_used_says_which_and_why(spans, message):
    with pytest.raises(LS.LookSpanError) as e:
        norm(spans)
    assert str(e.value).startswith(message), str(e.value)


def test_no_spans_is_none():
    assert norm(None) == []
