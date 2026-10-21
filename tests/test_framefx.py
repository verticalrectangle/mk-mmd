"""Frame effects (mkmmd.core.framefx: the [[fx]] entries checked and timed; mkmmd.framefx: the picture work) on synthetic
pictures: their errors, that they may overlap anything and stack in table order, which layers they ask for (points through
the shot in the cut, the frames an echo reads), their strength keys, and what each effect does to a picture."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
MOON = PAL.get("rose-pine-moon")
W, H = 64, 36


def project(fx=(), **extra):
    return {"project": {"duration": 4.0}, "output": [{"name": "t", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 2.0}, {"name": "b", "from": 2.0, "to": 4.0}],
            "fx": list(fx), **extra}


def plan(fx=(), **extra):
    return TR.plan(project(fx, **extra), FPS, F0, MOON)


def ramp():
    """A picture whose red is its column and green its row (0..1): where a pixel came from can be read off it."""
    xx, yy = np.meshgrid(np.linspace(0, 1, W, dtype=np.float32), np.linspace(0, 1, H, dtype=np.float32))
    return np.dstack([xx, yy, np.zeros_like(xx)])


def runner(pl, frames=None, points=None):
    return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: (frames or {}).get(g),
                      point_of=lambda key, g: (points or {})[(key, g)])


# ---------------------------------------------------------------- the entries
@pytest.mark.parametrize("spec, frag", [
    ({"kind": "blur", "from": 0, "to": 1}, "none of lens"),
    ({"kind": "lens", "from": 0, "to": 1}, "`center` is required"),
    ({"kind": "lens", "from": 1, "to": 0.5, "center": [0.5, 0.5]}, "must come after"),
    ({"kind": "lens", "from": 0, "to": 1, "center": [0.5, 0.5], "radius_m": 0.5}, "scene point's depth"),
    ({"kind": "lens", "from": 0, "to": 1, "center": {"point": "x"}, "radius": 0.1, "radius_m": 0.5}, "not both"),
    ({"kind": "push", "from": 0, "to": 1, "center": [0.5, 0.5], "strength": 0.5}, "unknown keys ['strength']"),
    ({"kind": "echo", "from": 0, "to": 1, "blend": "screen"}, "none of lighten"),
    ({"kind": "slitscan", "from": 0, "to": 1, "now": "middle"}, "the edge that shows the present"),
    ({"kind": "scope", "from": 0, "to": 1, "stem": "piano"}, "none of drums"),
    ({"kind": "echo", "from": 0, "to": 1, "strength": [[0, 1], [0, 2]]}, "two keys at the same second"),
    ({"kind": "echo", "from": 3.5, "to": 4.5}, "outside the clip"),
    ({"kind": "push", "from": 3, "to": 4, "center": [0.5, 0.5], "inside": 3.5, "play": True}, "run past the clip"),
    ({"kind": "zoom", "from": 0, "to": 1}, "give `times`"),
    ({"kind": "zoom", "from": 0, "to": 1, "times": [0.5], "lane": "kick"}, "one of them"),
    ({"kind": "shock", "from": 0, "to": 1, "times": [0.5], "min_db": -6}, "pick from a lane"),
    ({"kind": "slab", "from": 0, "to": 0.2, "in": 0.15, "out": 0.15}, "longer than the slab is on"),
    ({"kind": "slab", "from": 0, "to": 1, "dir": "sideways"}, "the way it travels"),
    ({"kind": "halftone", "from": 0, "to": 1, "cell": 1}, "cell = 1: expected at least 2"),
    ({"kind": "halftone", "from": 0, "to": 1, "offset": 3}, "expected [x, y]"),
    ({"kind": "split", "from": 0, "to": 1, "bands": 2.5}, "a whole number"),
    ({"kind": "split", "from": 0, "to": 1, "hold": 0}, "hold = 0"),
    ({"kind": "mosh", "from": 0, "to": 1, "block": 0.5}, "block = 0.5"),
])
def test_bad_fx_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError, match=frag.replace("[", r"\[").replace("]", r"\]")):
        plan([spec])


def test_effects_may_overlap_each_other_and_a_transition_and_stack_in_table_order():
    lenses = [{"kind": "lens", "from": 1.0, "to": 3.0, "center": [0.3, 0.5], "radius": 0.2},
              {"kind": "lens", "from": 1.5, "to": 2.5, "center": [0.7, 0.4], "radius": 0.3}]
    pl = plan(lenses, transition=[{"at": 2.0, "kind": "slash", "dur": 0.5}])          # no overlap error
    f = F0 + 60
    img = ramp()
    got = runner(pl).frame(f, img)
    one, two = [(fx["center"]["at"][0] * W - 0.5, fx["center"]["at"][1] * H - 0.5, fx["radius"] * H) for fx in pl["fx"]]
    want = FF.lens(FF.lens(img, one[:2], one[2]), two[:2], two[2])
    assert np.allclose(got, want) and not np.allclose(got, FF.lens(FF.lens(img, two[:2], two[2]), one[:2], one[2]))
    assert not runner(pl).active(F0 + 29) and runner(pl).frame(F0 + 29, img) is img      # before the first one


def test_a_scene_centre_is_projected_through_the_shot_in_the_cut_on_each_frame():
    pl = plan([{"kind": "lens", "from": 1.9, "to": 2.1, "center": {"point": "obj('hole').loc"}, "radius_m": 0.5}])
    dem = TR.demands(pl)
    shots = {f: [it["shot"] for it in items if it["kind"] == "point"] for f, items in dem.items()}
    assert shots == {F0 + 57: ["a"], F0 + 58: ["a"], F0 + 59: ["a"], F0 + 60: ["b"], F0 + 61: ["b"], F0 + 62: ["b"]}


def test_look_renders_the_frames_an_echo_and_a_push_read_and_none_before_the_clip():
    pl = plan([{"kind": "echo", "from": 0.0, "to": 1.0, "count": 3, "every": 2},
               {"kind": "push", "from": 3.0, "to": 3.5, "center": [0.5, 0.5], "inside": 0.5}])
    assert TR.fx_frames(pl, [F0 + 3]) == {F0 + 1: [{"kind": "plate", "shot": "a"}]}      # F0 - 1, F0 - 3 do not exist
    assert TR.fx_frames(pl, [F0 + 95]) == {F0 + 15: [{"kind": "plate", "shot": "a"}]}    # a still, frame 0.5 s


def test_strength_is_linear_between_keys_and_held_outside_them():
    fx = FX.normalize({"kind": "echo", "from": 0, "to": 4, "strength": [[2.0, 1.0], [1.0, 0.0]]}, MOON)
    assert [FX.strength(fx, t) for t in (0.0, 1.0, 1.25, 2.0, 3.0)] == pytest.approx([0.0, 0.0, 0.25, 1.0, 1.0])


# ---------------------------------------------------------------- the pictures
def test_a_lens_shows_each_pixel_the_picture_from_closer_in_by_theta_squared_over_r():
    img, c, theta = ramp(), (31.5, 17.5), 6.0
    out = FF.lens(img, c, theta)
    for r in (9.0, 12.0, 20.0):                                        # outside the ring: pulled in from r - theta^2 / r
        x = c[0] + r
        assert out[17, int(x)][0] == pytest.approx((int(x) - theta ** 2 / (int(x) - c[0])) / (W - 1), abs=2e-3)
    assert out[17, 34][0] < c[0] / (W - 1)                           # inside the ring: the far side, folded in
    assert np.all(FF.lens(img, c, theta, shadow=0.5)[17, 30:34] == 0)  # the shadow


def test_a_push_starts_as_the_picture_grows_its_disc_and_ends_as_the_inside_frame():
    rng = np.random.default_rng(1)
    img, inner = rng.uniform(0, 1, (H, W, 3)).astype(np.float32), rng.uniform(0, 1, (H, W, 3)).astype(np.float32)
    first, last = FF.push(img, inner, (20.0, 10.0), 2.0, 0.0), FF.push(img, inner, (20.0, 10.0), 2.0, 1.0)
    yy, xx = np.mgrid[0:H, 0:W]
    away = np.hypot(xx - 20, yy - 10) > 3.5
    assert np.allclose(first[away], img[away]) and np.allclose(last, inner, atol=1e-5)
    flat0, flat1 = np.zeros((H, W, 3), np.float32), np.ones((H, W, 3), np.float32)       # the inside is all 1
    cover = [float(FF.push(flat0, flat1, (20.0, 10.0), 2.0, u).mean()) for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert cover[0] < 0.01 and cover[-1] == pytest.approx(1.0) and all(a < b for a, b in zip(cover, cover[1:]))


def test_an_echo_never_touches_the_ground_and_tints_the_figure():
    dark, fig = np.zeros((H, W, 3), np.float32), np.zeros((H, W, 3), np.float32)
    fig[10:20, 30:34] = 1.0
    lit = FF.echo(dark, [(dark, 0.9), (fig, 0.5)], "lighten", tint=[0.2, 0.6, 1.0])
    assert lit[0, 0].tolist() == [0, 0, 0] and lit[15, 31] == pytest.approx([0.1, 0.3, 0.5])
    light, dfig = np.ones((H, W, 3), np.float32), np.ones((H, W, 3), np.float32)
    dfig[10:20, 30:34] = 0.0
    shade = FF.echo(light, [(light, 0.9), (dfig, 1.0)], "darken", tint=[0.2, 0.6, 1.0])
    assert shade[0, 0].tolist() == [1, 1, 1] and shade[15, 31] == pytest.approx([0.2, 0.6, 1.0])


def test_a_slitscan_shows_the_present_at_its_edge_and_strength_scales_its_depth():
    frames = {g: np.full((H, W, 3), (F0 + 45 - g) / 100.0, np.float32) for g in range(F0, F0 + 46)}  # value: 45 - its frame
    pl = plan([{"kind": "slitscan", "from": 0.0, "to": 2.0, "depth": 20, "now": "left",
                "strength": [[0.0, 1.0], [1.0, 1.0], [1.5, 0.5]]}])
    full = runner(pl, frames).frame(F0 + 30, frames[F0 + 30])                              # 1.0 s: 20 frames deep
    assert full[0, 0, 0] == pytest.approx(0.15) and full[0, -1, 0] == pytest.approx(0.35)  # now, and 20 frames back
    half = runner(pl, frames).frame(F0 + 45, frames[F0 + 45])                              # 1.5 s: strength 0.5, 10 deep
    assert half[0, 0, 0] == 0.0 and half[0, -1, 0] == pytest.approx(0.10)
    assert np.all(np.diff(half[0, :, 0]) >= 0)                                             # older toward the far edge


def test_a_scope_of_a_sine_against_itself_a_quarter_period_later_is_a_circle():
    sr, hz = 44100, 50.0
    x = np.sin(2 * np.pi * hz * np.arange(sr) / sr).astype(np.float32)
    pts = FF.scope_points(x, sr // 2, int(sr / hz), int(round(sr / hz / 4)), 0.8)
    assert np.hypot(pts[:, 0], pts[:, 1]) == pytest.approx(np.full(len(pts), 0.8), abs=2e-3)
    img = FF.draw_trace(np.zeros((H, W, 3), np.float32), pts, (32.0, 18.0), 12.0, [1.0, 0.5, 0.0], 1.0, 0.0)
    lit = np.argwhere(img[..., 0] > 0.3)
    r = np.hypot(lit[:, 1] - 32.0, lit[:, 0] - 18.0)
    assert r.min() > 12.0 * 0.8 - 1.5 and r.max() < 12.0 * 0.8 + 1.5 and np.all(img[..., 2] == 0)


def test_a_zoom_punches_in_about_its_centre_on_each_hit_and_eases_back_from_the_strongest():
    hits = {"kick": [{"t": 0.5, "db": 0.0}, {"t": 1.0, "db": -20.0}, {"t": 1.5, "db": -1.0}]}
    pl = plan([{"kind": "zoom", "from": 0.0, "to": 2.0, "lane": "kick", "min_db": -6, "amount": 0.1, "attack": 0.0,
                "decay": 0.2, "center": [0.25, 0.5]}])
    fx = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, hits=hits)
    img = ramp()
    assert fx.frame(F0 + 14, img) is img or np.array_equal(fx.frame(F0 + 14, img), img)       # before the first hit
    on = fx.frame(F0 + 15, img)                                                               # 0.5 s: in by 10 %
    c = (0.25 * W - 0.5, 0.5 * H - 0.5)
    x = 50
    assert on[18, x, 0] == pytest.approx(img[18, int(round(c[0] + (x - c[0]) / 1.1)), 0], abs=0.02)
    assert np.array_equal(fx.frame(F0 + 30, img)[18], FF.scale_about(img, c, 1.0 + 0.1 * np.exp(-0.5 / 0.2))[18])  # 1.0: too quiet
    assert FX.punch([0.5, 0.7], [0.5], 0.0, 0.2) == pytest.approx([1.0, np.exp(-1.0)])


def test_a_shock_lights_its_front_where_it_has_run_and_leaves_the_rest_of_the_picture():
    pl = plan([{"kind": "shock", "from": 0.0, "to": 2.0, "times": [1.0], "speed": 0.5, "width": 0.05, "amount": 0.05,
                "life": 0.6, "glow": 1.0, "color": [1.0, 0.0, 0.0], "center": [0.5, 0.5]}])
    flat = np.zeros((H, W, 3), np.float32)
    out = runner(pl).frame(F0 + 36, flat)                                  # 0.2 s on: the front 0.1 heights out
    r = 0.5 * 0.2 * H
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - (W / 2 - 0.5), yy - (H / 2 - 0.5))
    assert out[..., 0][np.abs(d - r) < 0.5].min() > 0.4                   # the front is lit ...
    assert out[..., 0][d > r + 6].max() < 1e-3 and np.all(out[..., 1:] == 0)   # ... nothing beyond it, in its colour
    assert np.array_equal(runner(pl).frame(F0 + 29, flat), flat)           # before the hit


def test_a_slab_comes_in_from_behind_rests_at_its_place_and_goes_off_the_far_side():
    pl = plan([{"kind": "slab", "from": 1.0, "to": 2.0, "at": [0.5, 0.5], "size": [0.6, 0.2], "slant": 0.0,
                "angle": 0.0, "color": [1.0, 1.0, 1.0], "dir": "right", "in": 0.2, "out": 0.2, "shutter": 0.0}])
    black = np.zeros((H, W, 3), np.float32)
    run = runner(pl)
    rest = run.frame(F0 + 45, black)[..., 0]                               # 1.5 s: at rest, centred
    cols = np.flatnonzero(rest[18] > 0.5)
    assert cols.min() == pytest.approx(W / 2 - 0.3 * H, abs=1) and cols.max() == pytest.approx(W / 2 + 0.3 * H, abs=1)
    assert run.frame(F0 + 30, black)[..., 0].max() == 0.0                  # 1.0 s: still off the left side
    coming = run.frame(F0 + 33, black)[..., 0]                             # on its way in, from the left
    assert 0 < np.flatnonzero(coming[18] > 0.5).max() < cols.max()
    going = run.frame(F0 + 58, black)[..., 0]                              # on its way out, to the right
    assert np.flatnonzero(going[18] > 0.5).min() > cols.min()
    lean = FF.slab_cover((W, H), [(31.5, 17.5)], 20.0, 10.0, 30.0, 0.0)  # italics: the top stands to the right
    top, bottom = np.flatnonzero(lean[13] > 0.5), np.flatnonzero(lean[22] > 0.5)
    assert top.mean() - bottom.mean() == pytest.approx(9 * np.tan(np.radians(30.0)), abs=1.0)
    blurred = FF.slab_cover((W, H), [(20.0, 17.5), (26.0, 17.5)], 10.0, 10.0, 0.0, 0.0)
    assert blurred[17, 23] == pytest.approx(1.0) and blurred[17, 17] == pytest.approx(0.5)   # both places, half each


def test_a_tint_maps_dark_to_dark_mid_grey_to_its_colour_and_light_to_light_by_its_strength():
    pl = plan([{"kind": "tint", "from": 0.0, "to": 2.0, "color": [0.2, 0.9, 0.4], "strength": [[0.0, 1.0], [1.0, 0.0]]}])
    img = np.zeros((H, W, 3), np.float32)
    img[:, :20], img[:, 20:40], img[:, 40:] = 0.0, 0.5, 1.0
    full = runner(pl).frame(F0, img)
    assert full[0, 0] == pytest.approx([0, 0, 0]) and full[0, 30] == pytest.approx([0.2, 0.9, 0.4], abs=1e-3)
    assert full[0, 50] == pytest.approx([1, 1, 1], abs=1e-5)
    assert np.array_equal(runner(pl).frame(F0 + 30, img), img)             # strength 0: untouched


def test_a_halftone_screens_the_shadows_in_dots_that_grow_with_darkness_and_prints_its_plates_off_register():
    img = np.ones((H, W, 3), np.float32)
    img[:, :16], img[:, 16:32], img[:, 32:48] = 0.05, 0.25, 0.45      # near black, dark, just under the level; light
    out = FF.halftone(img, 6.0, 45.0, 0.5, [0.0, 0.0, 0.0], 0.4, (0.0, 0.0))
    ink = [float((out[:, a:b].mean(-1) < 0.15).mean()) for a, b in ((0, 16), (16, 32), (32, 48))]
    assert ink[0] > ink[1] > ink[2] > 0.0 and np.array_equal(out[:, 48:], img[:, 48:])   # the lights stay clean
    ground = 0.25 + 0.75 * 0.4 * 0.5                                       # lifted toward paper by lift x darkness
    assert out[4:30, 18:30].min() < 0.05 and out[4:30, 18:30].max() == pytest.approx(ground, abs=1e-5)   # dots on it
    off = FF.halftone(img, 6.0, 45.0, 0.5, [0.0, 0.0, 0.0], 0.4, (3.0, 0.0))
    assert np.allclose(off[:, 10:58, 0], out[:, 7:55, 0], atol=1e-5) and np.allclose(off[..., 1], out[..., 1])
    assert np.allclose(off[:, 10:58, 2], out[:, 13:61, 2], atol=1e-5)          # red 3 px one way, blue the other


def test_a_split_tears_bands_sideways_holds_each_tear_and_breaks_again_after_its_hold():
    pl = plan([{"kind": "split", "from": 1.0, "to": 2.0, "bands": 8, "shift": 0.3, "share": 1.0, "rgb": 0.0,
                "hold": 0.1, "seed": 3}])
    img = ramp()
    a, b, c = runner(pl).frame(F0 + 30, img), runner(pl).frame(F0 + 31, img), runner(pl).frame(F0 + 33, img)
    assert np.array_equal(a, b) and not np.array_equal(a, c)               # one tear for 0.1 s, then another
    torn = [r for r in range(H) if not np.array_equal(a[r], img[r])]
    assert torn and all(any(np.array_equal(a[r], np.roll(img[r], dx, 0)) for dx in range(1, W)) for r in torn)
    planes = FF.split(img, np.random.default_rng(0), 1, 0.0, 0.0, 6.0)     # no tear: the planes 2 px apart everywhere
    assert np.array_equal(planes[..., 0], np.roll(img[..., 0], 2, 1))
    assert np.array_equal(planes[..., 2], np.roll(img[..., 2], -2, 1))


def test_a_mosh_carries_the_frame_before_the_cut_along_the_new_shots_motion_from_any_frame_asked():
    import cv2
    w, h = 320, 180
    tex = cv2.GaussianBlur(np.random.default_rng(1).random((h, w)).astype(np.float32), (0, 0), 3)
    tex = np.repeat(((tex - tex.min()) / (tex.max() - tex.min()))[..., None], 3, 2)
    before = np.zeros((h, w, 3), np.float32)
    before[60:120, 130:190] = 1.0                                          # shot a's last frame: a white block
    frames = {F0 + 59: before, **{F0 + 60 + k: np.roll(tex, 4 * k, 1) for k in range(8)}}   # shot b pans right
    pl = plan([{"kind": "mosh", "from": 2.0, "to": 2.5, "block": 8}])
    fx = FF.FrameFx(pl["fx"], (w, h), fps=FPS, frame0=F0, frame_of=frames.get, point_of=None)
    left = [int(np.argmax(fx.frame(f, frames[f])[90, :, 0] > 0.5)) for f in range(F0 + 60, F0 + 66)]
    assert left == [130 + 4 * k for k in range(6)]                         # held at the cut, then moved with shot b
    late = FF.FrameFx(pl["fx"], (w, h), fps=FPS, frame0=F0, frame_of=frames.get, point_of=None)
    assert np.allclose(late.frame(F0 + 65, frames[F0 + 65]), fx.frame(F0 + 65, frames[F0 + 65]), atol=1e-4)
    assert FX.sources(pl["fx"][0], F0 + 63, F0) == list(range(F0 + 59, F0 + 64))
