"""A time range that runs to the clip's end (`t=A:<duration>`) ends on the clip's last frame, not one past it."""
from mkmmd.core import frames as FR


def test_a_time_range_to_the_clips_end_stops_on_its_last_frame():
    got = FR.parse("t=25.9:26.05", fps=60, frame0=181, duration=26.05)
    assert got[-1] == 181 + 1563 - 1 and got[0] == 181 + 1554
