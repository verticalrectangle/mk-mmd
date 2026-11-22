"""The paper dragonfly, bpy-free (mkmmd.core.dragonfly): its meshes at unit span, the wings' driver expressions against
the numpy whirr, the whirr read on a stepped grid, and the options. The flight and the wire are tests/test_flight.py;
the Blender half is tests/test_flight_build.py."""
import math

import numpy as np
import pytest

from mkmmd.core import dragonfly as DF

BLENDER = {"radians": math.radians, "cos": math.cos}


def test_parts_are_closed_and_the_span_is_one():
    P = DF.parts()
    for key, m in P.items():
        assert m.is_closed(), key
        assert m.volume() > 0, key
    tips = []
    for wing in ("fore", "hind"):
        for side, sx in (("L", 1.0), ("R", -1.0)):
            V = P[f"{wing}_{side}"].V + np.array(DF.hinge(wing, side))
            tips.append(sx * V[:, 0].max() if side == "L" else -V[:, 0].min())
    assert max(tips) == pytest.approx(0.5, abs=0.01)                 # wing tip to wing tip, flat: the unit span
    lo, hi = P["body"].bbox()
    assert 0.62 <= hi[1] - lo[1] <= 0.75                             # a dragonfly's body is about its span long
    eye = DF.EYE_R / (hi[0] - lo[0])
    assert eye > 0.25                                                # big eyes: a toy's, not a specimen's


@pytest.mark.parametrize("rest", [0.0, 0.4, 1.0])
def test_wing_drivers_are_the_whirr(rest):
    ex = DF.expressions()
    P = {"rest": rest, "amp": 70.0, "mid": 10.0, "lag": 0.5, "droop": 4.0}
    for ph in (0.0, 0.13, 0.41, 0.77):
        env = dict(BLENDER, phase=ph, **P)
        fore = float(DF.wing_angle(ph, rest, P["amp"], P["mid"], P["droop"]))
        hind = float(DF.wing_angle(ph + P["lag"], rest, P["amp"], P["mid"], P["droop"]))
        assert eval(ex["fore_L"], {}, env) == pytest.approx(-fore)    # the left wing lifts by turning -Y
        assert eval(ex["fore_R"], {}, env) == pytest.approx(fore)
        assert eval(ex["hind_L"], {}, env) == pytest.approx(-hind)
    assert all(len(e) <= 255 for e in ex.values())                   # Blender keeps no longer simple expression


@pytest.mark.parametrize("step", [0.0, 12.0])
def test_the_whirr_reads_on_every_pose(step):
    """A wing stands somewhere else on each pose at 60 a second and on the set's 12 a second grid alike (a whirr
    counted in beats a second would look still when stepped)."""
    frames = np.arange(181, 361)
    k = DF.poses(frames, 181, 60.0, step, 3.03)
    shown = np.unique(k)
    ang = np.degrees(DF.wing_angle(shown * DF.DEFAULTS["whirr"], 0.0, 70.0, 10.0, 4.0))
    jump = np.abs(np.diff(ang))
    assert np.median(jump) > 25.0
    if step:
        held = np.diff(k)
        assert set(np.unique(held)) <= {0.0, 1.0} and (held == 0).sum() > 0.7 * len(held)     # held between poses


def test_options_refuse_what_they_cannot_build():
    with pytest.raises(ValueError, match="whirr"):
        DF.options({"whirr": 0.9})
    with pytest.raises(ValueError, match="hand"):
        DF.options({"hand": [1.0, 2.0]})
    with pytest.raises(ValueError, match="inks"):
        DF.options({"inks": "foam"})
