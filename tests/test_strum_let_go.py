"""A pick hand letting go of the strings (mkmmd.core.strum.let_go, clashes): the spans it is off them, and the strum
windows that would key it there."""
import math

import pytest

from mkmmd.core import strum as SM


def test_spans_run_from_the_grip_key_before_to_the_one_after():
    keys = [{"t": 1.0, "grip": True}, {"t": 1.2, "at": [0, 0, 1]}, {"t": 1.5, "at": [0, 0, 1.1]},
            {"t": 2.0, "grip": True}, {"t": 3.0, "grip": True}, {"t": 3.3, "at": [0, 0, 1]}]
    assert SM.let_go(keys) == [(1.0, 2.0), (3.0, math.inf)]


@pytest.mark.parametrize("keys", [
    [{"t": 1.0, "at": [0, 0, 1]}],                                   # leaves from nowhere: drifts from the settle
    [{"t": 1.0, "grip": True, "at": [0, 0, 1]}],                     # both
    [{"t": 1.0}],                                                    # neither
    [{"t": 1.0, "grip": True}, {"t": 0.5, "at": [0, 0, 1]}],         # out of order
])
def test_bad_keys_are_refused(keys):
    with pytest.raises(SM.StrumError):
        SM.let_go(keys)


def test_a_strum_window_over_a_let_go_span_clashes():
    spans = [(23.9, math.inf)]
    assert SM.clashes(spans, (0.0, 23.85)) == []
    assert SM.clashes(spans, (0.0, 24.1)) == spans
