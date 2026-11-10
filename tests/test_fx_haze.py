"""The haze ([[fx]] kinds "mist", "fog" and "bokeh", mkmmd.core.haze and mkmmd.haze): what the keys refuse, a mist that
only spreads the light over its threshold, a fog thicker toward the ground and kept behind the figure but for its veil,
a bokeh whose lights pass behind her but for the largest and swell on a hit, the same frame in any order, and the GPU
leaving the kinds to the CPU."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import postgpu as PG
from mkmmd.core import framefx as FX
from mkmmd.core import haze as HZ
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
MOON = PAL.get("rose-pine-moon")
W, H = 160, 90


def entry(kind, **kw):
    return {"kind": kind, "from": 0.0, "to": 3.0, **kw}


def plan(*fx):
    data = {"project": {"duration": 4.0}, "output": [{"name": "t", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, MOON)


def runner(pl, figure=None):
    return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                      figure_of=(lambda key, g: figure) if figure is not None else None)


def her():
    m = np.zeros((H, W), np.float32)
    m[20:70, 60:100] = 1.0
    return m


@pytest.mark.parametrize("spec, frag", [
    (entry("mist", radius=0.9), "radius = 0.9: expected 0.002..0.5"),
    (entry("mist", blend="over"), "blend 'over' is none of screen, add"),
    (entry("fog", colors=["rose"] * 5), "one to four colours"),
    (entry("fog", mask="ground"), "needs the figure's `objects`"),
    (entry("bokeh", size=[0.1, 0.02]), "the smallest radius first"),
    (entry("bokeh", every=2), "give the `lane`"),
    (entry("bokeh", lights=3), "unknown keys ['lights']"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError) as e:
        plan(spec)
    assert frag in str(e.value)


def test_a_mist_spreads_only_the_light_over_its_threshold():
    img = np.full((H, W, 3), 0.3, np.float32)
    r = runner(plan(entry("mist", radius=0.06, threshold=0.7, knee=0.1)))
    assert np.array_equal(r.frame(F0 + 5, img), img)             # nothing over the threshold: the picture as it was
    img[40:50, 75:85] = 1.0                                      # a light
    out = r.frame(F0 + 5, img)
    assert (out >= img - 1e-6).all() and out.max() <= 1.0
    glow = out[45, :, 0] - img[45, :, 0]
    assert glow[70] > glow[60] > glow[40] >= 0.0 and glow[70] > 0.02 and glow[5] < 0.01     # round it, dying with distance
    soft = runner(plan(entry("mist", radius=0.06, threshold=0.99, diffuse=1.0))).frame(F0 + 5, img)
    assert soft[45, 80, 0] < 1.0 and soft[45, 72, 0] > 0.3       # the diffusion: the light's edge gone soft


def test_a_fog_is_thicker_at_the_ground_drifts_and_stays_behind_her():
    img = np.zeros((H, W, 3), np.float32)
    fog = entry("fog", colors="text", density=1.0, floor=1.0, speed=0.2)
    r = runner(plan(fog))
    a, b = r.frame(F0 + 10, img), r.frame(F0 + 40, img)
    assert a[-20:].mean() > 3.0 * a[:20].mean() > 0.0            # thick at the ground, thin at the top
    assert not np.allclose(a, b)                                 # it drifts
    assert np.array_equal(runner(plan(fog)).frame(F0 + 40, img), b)          # a frame is the same wherever a render starts
    m = her()
    behind = runner(plan(dict(fog, mask="ground", objects="Reisen*", veil=0.0)), m).frame(F0 + 10, img)
    assert np.array_equal(behind[20:70, 60:100], img[20:70, 60:100]) and np.array_equal(behind[80:], a[80:])
    veiled = runner(plan(dict(fog, mask="ground", objects="Reisen*", veil=0.5)), m).frame(F0 + 10, img)
    assert np.allclose(veiled[20:70, 60:100], 0.5 * a[20:70, 60:100], atol=1e-6)


def test_a_bokeh_passes_behind_her_but_for_its_largest_lights_and_swells_on_a_hit():
    spec = entry("bokeh", count=60, size=[0.03, 0.1], colors="text", twinkle=0.0, speed=0.05, seed=3)
    pl = plan(spec)
    lights = HZ.lights(pl["fx"][0], 1.0, W / H)
    assert len(lights["r"]) == 60 and lights["front"].sum() == 9 and (np.diff(lights["r"]) >= 0).all()
    assert lights["r"][~lights["front"]].max() <= lights["r"][lights["front"]].min()        # the largest are in front
    img = np.zeros((H, W, 3), np.float32)
    plain = runner(pl).frame(F0 + 30, img)
    assert plain.max() > 0.2 and (plain >= 0.0).all()
    m = np.ones((H, W), np.float32)                              # she fills the frame: only the front lights show
    masked = runner(plan(dict(spec, mask="ground", objects="Reisen*")), m).frame(F0 + 30, img)
    assert 0.0 < masked.sum() < plain.sum()
    none = runner(plan(dict(spec, mask="ground", objects="Reisen*", front=0.0)), m).frame(F0 + 30, img)
    assert np.array_equal(none, img)
    hit = runner(plan(dict(spec, times=[1.0], pulse=1.0, decay=0.2)))
    before, on = hit.frame(F0 + 29, img), hit.frame(F0 + 30, img)
    assert on.sum() > 1.3 * before.sum() and np.array_equal(runner(pl).frame(F0 + 29, img), before)


def test_the_gpu_leaves_the_haze_to_the_cpu():
    for kind in HZ.KINDS:
        r = runner(plan(entry(kind)))
        assert PG.ops(r, r.fxs[0], F0 + 3) is None
