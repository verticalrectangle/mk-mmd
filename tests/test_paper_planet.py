"""The paper planet, bpy-free (mkmmd.core.paper_planet): its meshes and the flat landing spot, and its motion: in place
on `up`, a landing that sinks from its own time, bobs and settles lower, the landing spot riding it."""
import math

import numpy as np
import pytest

from mkmmd.core import paper_planet as PP


def test_parts_are_closed_and_the_top_is_a_landing_spot():
    P = PP.parts()
    for key in ("ball", "craters", "skewer", "pick", "flag"):
        assert P[key].is_closed(), key
        assert P[key].volume() > 0, key
    V = P["ball"].V
    flat = V[np.abs(V[:, 2] - PP.Z_TOP) < 1e-9]
    assert np.hypot(flat[:, 0], flat[:, 1]).max() == pytest.approx(PP.R * math.sin(PP.THETA_CAP), rel=0.03)
    assert V[:, 2].max() == pytest.approx(PP.Z_TOP)                    # nothing stands above the flat top
    assert PP.R * math.sin(PP.THETA_CAP) * 0.8 > 0.12                  # wide enough for two feet at 0.8 m
    craters = PP.crater_spots()
    o, a, _x = PP.pick_frame()
    for d, r in craters:                                               # cut-paper craters clear of the spot and the flag
        assert math.acos(min(1.0, d[2])) > PP.THETA_CAP + r / PP.R
        assert math.acos(float(np.clip(d @ PP.flag_point(), -1, 1))) > r / PP.R
    assert len(np.unique(P["flag"].Qm)) == len(PP.STRIPES)            # striped


def test_rises_on_up_and_a_landing_bobs_and_settles():
    O = PP.options({"up": 17.0, "land": 17.26, "rise": 0.7, "depth": 1.6})
    t = np.arange(15.0 * 60, 19.0 * 60 + 1) / 60.0
    m = PP.motion(O, t)
    z = m["z"]
    i = {s: int(round((s - 15.0) * 60)) for s in (16.2, 17.0, 17.26, 18.9)}
    assert z[i[16.2]] == pytest.approx(-1.6)                            # still below before it sets off
    assert z[i[17.0]] == pytest.approx(0.0, abs=1e-9)                   # in place ON up
    land = i[17.26]
    assert z[land + 3] < z[land] - 0.01                                 # a landing sinks from its own frame...
    assert np.diff(np.sign(np.diff(z[land:land + 60]))).any()          # ...bobs...
    assert z[i[18.9]] == pytest.approx(-O["settle"], abs=0.004)         # ...and settles lower
    assert abs(m["whip"][land + 6]) > 0.3 and abs(m["whip"][i[16.2]]) < 0.2     # the flag whips on it
    top = PP.top(O, t)
    assert top[:, 2] - z == pytest.approx(np.full(len(t), PP.Z_TOP * O["size"]), abs=0.002)   # the spot rides it


def test_options_refuse_what_they_cannot_build():
    with pytest.raises(ValueError, match="land"):
        PP.options({"land": "17.26"})
    with pytest.raises(ValueError, match="stripes"):
        PP.options({"stripes": []})
