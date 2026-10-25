"""Film ([[fx]] kind "film", mkmmd.core.picturefx and mkmmd.picturefx): its keys, the held frames (`fps`: which clip
frame each film frame comes from, the frames it reads), that a held frame is the picture as it reached the effect in
mk post's order and out of order as mk look asks, black and white, determinism, and scratches that persist and drift."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import picturefx as PF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import picturefx as PC
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 64, 36


def plan(*fx, fps=FPS):
    data = {"project": {"duration": 4.0}, "output": [{"name": "t", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}
    return TR.plan(data, fps, F0, DAWN)


def frames(n=130):
    """A clip whose frame g is a flat grey of its own (and a ramp, so the picture has detail)."""
    xx = np.linspace(0.0, 0.3, W, dtype=np.float32)[None, :, None]
    return {F0 + k: np.clip(np.full((H, W, 3), 0.2 + 0.004 * k, np.float32) + xx, 0, 1) for k in range(n)}


def runner(pl, clip, fps=FPS):
    return FF.FrameFx(pl["fx"], (W, H), fps=fps, frame0=F0, frame_of=clip.get, point_of=None)


@pytest.mark.parametrize("spec, frag", [
    ({"fps": 0}, "fps = 0"),
    ({"mono": "yes"}, "mono is true or false"),
    ({"grain": {"amount": 0.1, "grit": 2}}, "expected a number or {amount, size}"),
    ({"dust": {"size": 2}}, "expected a number or {count, size}"),
    ({"scratches": 40}, "expected 0..32"),
    ({"flicker": 1.5}, "expected 0..1"),
    ({"hair": -0.1}, "expected 0..1"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=frag.replace("{", r"\{").replace("}", r"\}")):
        FX.normalize({"kind": "film", "from": 0, "to": 1, **spec}, DAWN)


def test_numbers_or_tables():
    fx = FX.normalize({"kind": "film", "from": 0, "to": 1, "grain": 0.2, "dust": {"count": 5}}, DAWN)
    assert fx["grain"] == {"amount": 0.2, "size": PC.GRAIN_SIZE}
    assert fx["dust"] == {"count": 5.0, "size": PC.DUST_SIZE}


@pytest.mark.parametrize("film_fps, clip_fps, every", [(15, 30, 2), (10, 30, 3), (None, 30, 1), (60, 30, 1)])
def test_held_frames_change_fps_times_a_second(film_fps, clip_fps, every):
    fx = plan({"kind": "film", "from": 1.0, "to": 3.0, "fps": film_fps}, fps=clip_fps)["fx"][0]
    starts = [PC.film_step(fx, f) for f in range(fx["f0"], fx["f1"])]
    assert starts[0] == (0, fx["f0"])
    assert [g for _, g in starts] == [fx["f0"] + (i // every) * every for i in range(len(starts))]


def test_a_film_slower_than_no_multiple_of_the_clip():
    fx = plan({"kind": "film", "from": 1.0, "to": 3.0, "fps": 15}, fps=24)["fx"][0]      # 1.6 clip frames a film frame
    seen = {}
    for f in range(fx["f0"], fx["f1"]):
        k, g = PC.film_step(fx, f)
        assert fx["f0"] <= g <= f and PC.film_step(fx, g) == (k, g)
        seen.setdefault(k, []).append(f)
    assert sorted(len(v) for v in seen.values())[:1] == [1] and max(len(v) for v in seen.values()) == 2
    assert len(seen) == pytest.approx((fx["f1"] - fx["f0"]) / 1.6, abs=1)


def test_it_reads_the_first_frame_of_its_film_frame():
    pl = plan({"kind": "film", "from": 1.0, "to": 2.0, "fps": 10})
    fx = pl["fx"][0]
    f0 = fx["f0"]
    assert FX.sources(fx, f0, F0) == [f0] and FX.sources(fx, f0 + 2, F0) == [f0]
    assert TR.fx_frames(pl, [f0 + 2]) == {f0: [{"kind": "plate", "shot": "a"}]}
    assert TR.fx_frames(pl, [f0 + 3]) == {}                       # a film frame's first: the frame itself
    assert FX.holds(fx)


def test_a_held_frame_holds_in_order_and_out_of_order():
    pl = plan({"kind": "film", "from": 1.0, "to": 2.0, "fps": 10, "mono": True, "seed": 4})
    clip = frames()
    f0 = pl["fx"][0]["f0"]
    post = runner(pl, clip)
    a, b, c = (post.frame(f, clip[f]) for f in (f0, f0 + 1, f0 + 2))
    assert np.array_equal(a, b) and np.array_equal(a, c)         # three clip frames, one film frame
    d = post.frame(f0 + 3, clip[f0 + 3])
    assert not np.array_equal(a, d)
    look = runner(pl, clip)                                      # asked out of order, as mk look asks
    assert np.array_equal(look.frame(f0 + 2, clip[f0 + 2]), a)
    assert np.allclose(a[..., 0], a[..., 1]) and np.allclose(a[..., 1], a[..., 2])   # black and white


def test_the_effects_before_it_are_held_with_the_frame():
    tint = {"kind": "tint", "from": 1.0, "to": 2.0, "color": "love"}
    film = {"kind": "film", "from": 1.0, "to": 2.0, "fps": 10, "grain": 0, "dust": 0, "scratches": 0, "flicker": 0,
            "weave": 0, "vignette": 0}
    pl = plan(tint, film)
    clip = frames()
    f0 = pl["fx"][1]["f0"]
    post = runner(pl, clip)
    first = post.frame(f0, clip[f0])
    assert np.array_equal(post.frame(f0 + 1, clip[f0 + 1]), first)
    assert np.array_equal(runner(pl, clip).frame(f0 + 1, clip[f0 + 1]), first)   # the tint applied again to f0


def test_the_same_seed_draws_the_same_film():
    spec = {"kind": "film", "from": 1.0, "to": 2.0, "dust": 6, "scratches": 2, "hair": 1.0, "grain": 0.1}
    clip = frames()
    f = FX.frame_of(1.5, FPS, F0)
    one = runner(plan(spec), clip).frame(f, clip[f])
    assert np.array_equal(one, runner(plan(spec), clip).frame(f, clip[f]))
    assert not np.array_equal(one, runner(plan({**spec, "seed": 2}), clip).frame(f, clip[f]))


def test_scratches_persist_and_drift_a_little():
    fx = plan({"kind": "film", "from": 0.0, "to": 4.0, "scratches": 3, "seed": 5})["fx"][0]
    lives, drift = {}, 0.0
    for k in range(60):
        now, nxt = PC.film_scratches(fx, k), {s["id"]: s for s in PC.film_scratches(fx, k + 1)}
        assert now == PC.film_scratches(fx, k)
        for s in now:
            lives[s["id"]] = lives.get(s["id"], 0) + 1
            if s["id"] in nxt:
                drift = max(drift, abs(nxt[s["id"]]["x"] - s["x"]))
                assert nxt[s["id"]]["y"] == s["y"]
    assert len(lives) > 5 and min(lives.values()) >= 1 and np.median(list(lives.values())) >= 4   # a few frames each
    assert drift < 0.015                                         # ... drifting a little further sideways


def test_dust_and_hairs_are_drawn_as_light_and_dark():
    fx = plan({"kind": "film", "from": 0.0, "to": 4.0, "dust": {"count": 30, "size": 3}, "hair": 1.0,
               "scratches": 0})["fx"][0]
    k = next(k for k in range(200) if PC.film_hairs(fx, k))
    light, dark = PF.damage(fx, k, (320, 180), 1.0)
    assert light is not None and dark is not None
    assert light.max() > 0.5 and dark.max() > 0.5
    hair = PC.film_hairs(fx, k)[0]
    assert hair["points"].shape[1] == 2 and 0.05 < np.abs(np.diff(hair["points"], axis=0)).sum() < 0.6


def test_the_film_curve_is_an_s_through_mid_grey():
    x = np.linspace(0, 1, 11, dtype=np.float32)
    y = PF.film_curve(x, 1.0)
    assert y[0] == pytest.approx(0.0, abs=1e-6) and y[-1] == pytest.approx(1.0, abs=1e-6)
    assert y[5] == pytest.approx(0.5, abs=1e-6)
    assert y[2] < x[2] and y[8] > x[8]                            # darker shadows, brighter lights
    assert np.array_equal(PF.film_curve(x, 0.0), x)
