"""Freezes (hit-stops), bpy-free (mkmmd.core.freeze): `[[freeze]]` windows in Blender frames and the frame the world is
evaluated at on any frame. The Blender half (the cameras moved by their own keys over a held world, mkmmd.blender.freeze)
is checked by rendering and looking."""
import pytest

from mkmmd.core import freeze as FZ


def test_windows_are_frames_from_clip_seconds_sorted():
    assert FZ.normalize([{"from": 3.0, "to": 3.5}, {"from": 1.0, "to": 2.0}], 30, 121) == [[151, 181], [211, 226]]
    assert FZ.normalize([], 30, 121) == []


@pytest.mark.parametrize("specs, frag", [
    ([{"from": 1.0}], "expected {from, to}"),
    ([{"from": 1.0, "to": 2.0, "dur": 1}], "expected {from, to}"),
    ([{"from": 2.0, "to": 2.01}], "at least a frame"),
    ([{"from": 1.0, "to": 2.0}, {"from": 1.5, "to": 3.0}], "overlap")])
def test_bad_freezes_are_refused(specs, frag):
    with pytest.raises(FZ.FreezeError, match=frag):
        FZ.normalize(specs, 30, 121)


def test_the_world_holds_the_first_frame_of_a_window_and_runs_on_at_its_end():
    w = [[151, 181], [211, 226]]
    assert [FZ.source(w, f) for f in (150, 151, 165, 180, 181, 210, 211, 225, 226)] == \
        [150, 151, 151, 151, 181, 210, 211, 211, 226]
