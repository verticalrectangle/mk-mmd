"""Era looks ([[fx]] kinds digicam, chrome, gradmap, invert, testcard and cards: mkmmd.core.eralook and mkmmd.eralook)
and the era presets of film and camcorder: their keys, the regrade stepping on the cuts, the invert's flash safety, the
test card's layout, the cards' flights, and what each does to a picture."""
import math
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from mkmmd import eralook as ERF
from mkmmd import framefx as FF
from mkmmd import picturefx as PFX
from mkmmd.core import colorbars as CB
from mkmmd.core import eralook as ER
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
SIZE = (96, 54)


def entry(kind, **spec):
    return FX.normalize({"kind": kind, "from": 1.0, "to": 3.0, **spec}, DAWN)


def plan(*fxs, shots=None, size=SIZE):
    return TR.plan({"project": {"duration": 8.0}, "output": [{"name": "t", "size": list(size)}],
                    "shot": shots or [{"name": "a", "from": 0.0, "to": 8.0}], "fx": list(fxs)}, FPS, F0, DAWN)


def runner(pl, size=SIZE, figure=None, font_of=None):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                      figure_of=None if figure is None else (lambda key, g: figure), font_of=font_of)


def frame(t):
    return FX.frame_of(t, FPS, F0)


def system_font():
    """A TTF of this machine (a test card's labels need a real font), else the test is skipped."""
    if shutil.which("fc-match"):
        p = subprocess.run(["fc-match", "-f", "%{file}", "sans"], capture_output=True, text=True).stdout.strip()
        if p.lower().endswith((".ttf", ".otf")) and Path(p).is_file():
            return p
    for p in ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).is_file():
            return p
    pytest.skip("no TrueType font on this machine")


# ================================================================================================= keys
@pytest.mark.parametrize("kind, spec, frag", [
    ("digicam", {"jpeg": 0}, "jpeg = 0: expected 1..100"),
    ("digicam", {"date": "OCT 5"}, "expected a short stamp of digits"),
    ("digicam", {"corner": "middle"}, "none of bottom-right"),
    ("digicam", {"era": "2005"}, "unknown keys ['era']"),
    ("chrome", {"mask": "figure"}, "needs the figure's `objects`"),
    ("chrome", {"colors": []}, "expected a list of 1 to 8 colours"),
    ("gradmap", {"maps": "teal_lime"}, "expected a list of maps"),
    ("gradmap", {"maps": ["sunset"]}, "maps[0] = 'sunset' is none of teal_lime"),
    ("gradmap", {"maps": [["#000000"]]}, "2 to 8 colours"),
    ("invert", {"rate": 4}, "rate = 4: expected 0.1..3"),
    ("invert", {"to": 1.1}, "longer than the window"),
    ("testcard", {"bars": "smpte"}, "bars 'smpte' is none of dawn, true, both"),
    ("testcard", {"labels": ["x" * 40]}, "up to 4 short texts"),
    ("testcard", {"at": [1.2, 0.5]}, "in 0..1 of the frame"),
    ("cards", {}, "give `times`"),
    ("cards", {"times": [1.5], "path": "zigzag"}, "path 'zigzag' is none of in, across"),
    ("film", {"era": "1999"}, "era '1999' is none of 1967"),
    ("camcorder", {"era": 1967}, "era 1967 is none of 1985"),
])
def test_bad_entries_say_what_is_wrong(kind, spec, frag):
    with pytest.raises(FX.FxError, match=re.escape(frag)):
        entry(kind, **spec)


def test_every_era_look_is_a_frame_effect_with_a_rect():
    assert set(ER.KINDS) <= set(FX.KINDS)
    assert all({"rect", "feather", "strength"} <= FX.KEYS[k] for k in ER.KINDS)


# ================================================================================================= eras
def test_an_era_picks_a_preset_and_keys_beside_it_win():
    film = entry("film", era="1967")
    assert film["dye"] == 1.0 and film["fps"] == 24.0 and film["saturation"] > 1.0 and film["halation"] > 0.0
    assert entry("film", era=1967, fps=12)["fps"] == 12.0
    tape = entry("camcorder", era="1985")
    assert tape["switching"] > 0.0 and tape["tick"] and tape["date"] and tape["clock"] is not None
    plain = entry("film")
    assert (plain["dye"], plain["saturation"], plain["halation"]) == (0.0, 1.0, 0.0)
    assert entry("camcorder")["switching"] == 0.0


def test_a_dye_print_purifies_the_colours_and_halation_glows_round_the_lights():
    rng = np.random.default_rng(3)
    img = (0.3 + 0.4 * rng.random((40, 40, 3))).astype(np.float32)
    chroma = lambda x: float(np.mean(np.max(x, 2) - np.min(x, 2)))          # noqa: E731
    assert chroma(PFX.dye_print(img, 1.0, 1.35)) > chroma(img) * 1.2
    assert np.allclose(PFX.dye_print(img, 0.0, 1.0), img, atol=1e-6)
    dark = np.full((60, 60, 3), 0.1, np.float32)
    dark[25:35, 25:35] = 1.0
    glow = PFX.halation(dark, 0.6, 1.0)
    assert glow[30, 40, 0] > 0.15 > dark[30, 40, 0] and glow[30, 40, 0] > glow[30, 40, 2]   # a red-orange glow
    assert np.allclose(glow[2, 2], dark[2, 2], atol=0.02)                    # far from the light, none


def test_the_head_switching_tear_is_at_the_foot_of_the_tape():
    stripes = np.zeros((90, 160, 3), np.float32)
    stripes[:, ::8] = 1.0
    calm = {"tracking": 0.0, "noise": 0.0, "jitter": 0.0, "scanlines": 0.0, "chroma": 0.0, "soft": 0.0}
    a = entry("camcorder", **calm)
    b = entry("camcorder", switching=1.0, **calm)
    for fx in (a, b):
        fx["index"] = 0
    one, two = PFX.tape(stripes, a, 40, 1.3, 1.0), PFX.tape(stripes, b, 40, 1.3, 1.0)
    assert np.allclose(one[:80], two[:80], atol=1e-6)
    assert not np.allclose(one[-1], two[-1], atol=0.05)


# ================================================================================================= gradmap
def test_the_named_maps_run_dark_to_light():
    for colours in entry("gradmap", maps=list(ER.MAPS))["maps"]:
        lum = [CB.luminance(c) for c in colours]
        assert lum == sorted(lum)


def test_the_regrade_steps_on_every_cut_and_comes_round():
    shots = [{"name": n, "from": a, "to": b} for n, a, b in (("a", 0.0, 1.0), ("b", 1.0, 2.0), ("a2", 2.0, 3.0),
                                                             ("c", 3.0, 8.0))]
    fx = plan({"kind": "gradmap", "from": 0.5, "to": 4.0, "maps": ["teal_lime", "none", "blue_gold"]},
              shots=shots + [{"name": "plate", "plate": True}])["fx"][0]
    assert fx["cuts"] == [frame(1.0), frame(2.0), frame(3.0)]                # a plate is not a cut
    assert [ER.map_step(fx, frame(t), [], FPS, F0) for t in (0.6, 1.5, 2.5, 3.5)] == [0, 1, 2, 0]


def test_the_regrade_steps_on_hits_instead_when_it_has_them():
    fx = plan({"kind": "gradmap", "from": 0.0, "to": 4.0, "times": [1.0, 2.5], "offset": 1})["fx"][0]
    assert [ER.map_step(fx, frame(t), fx["times"], FPS, F0) for t in (0.5, 1.5, 3.0)] == [1, 2, 3]


def test_a_map_paints_luma_through_its_colours_and_none_leaves_the_picture():
    pl = plan({"kind": "gradmap", "from": 0.0, "to": 4.0, "maps": [["#000080", "#ffff00"], "none"], "times": [2.0]})
    img = np.zeros((54, 96, 3), np.float32)
    img[:, 48:] = 1.0
    ffx = runner(pl)
    out = ffx.frame(frame(1.0), img)
    assert np.allclose(out[0, 0], [0.0, 0.0, 128 / 255], atol=2e-3) and np.allclose(out[0, -1], [1.0, 1.0, 0.0])
    assert np.array_equal(ffx.frame(frame(3.0), img), img)


# ================================================================================================= invert
def test_thinned_hits_never_start_more_than_three_in_a_second():
    kept = ER.thin([i * 0.1 for i in range(40)], 3)
    assert ER.busiest(kept)[0] == 3 and len(kept) >= 10
    assert ER.thin([0.0, 0.5, 1.0], 3) == [0.0, 0.5, 1.0]


def test_a_window_eases_in_and_out_and_inverts_the_picture():
    pl = plan({"kind": "invert", "from": 1.0, "to": 2.0})
    fx = pl["fx"][0]
    assert ER.level(fx, 1.0) == 0.0 and ER.level(fx, 1.5) == 1.0 and 0.0 < ER.level(fx, 1.02) < 1.0
    assert 0.0 < ER.level(fx, 1.95) < 1.0
    img = np.full((54, 96, 3), 0.2, np.float32)
    assert np.allclose(runner(pl).frame(frame(1.5), img), 0.8, atol=1e-5)


def test_inverting_on_hits_seven_a_second_flashes_three_times_a_second_at_most():
    times = [1.0 + i * 0.15 for i in range(20)]                              # short pulses that would each flash
    pl = plan({"kind": "invert", "from": 1.0, "to": 4.0, "times": times, "in": 0.02, "hold": 0.03, "out": 0.05},
              size=(32, 18))
    img = np.full((18, 32, 3), 0.9, np.float32)
    ffx = runner(pl, (32, 18))
    frames = range(pl["fx"][0]["f0"], pl["fx"][0]["f1"])
    dark = [1.0 - float(ffx.frame(f, img).mean()) for f in frames]           # the light picture gone dark
    rises = ER.flashes(dark, [(f - F0) / FPS for f in frames])
    assert len(rises) >= 6 and ER.busiest(rises)[0] <= ER.FLASH_RATE


def test_strength_keys_that_flash_too_fast_are_refused():
    keys = [[1.0 + 0.1 * i, float(i % 2)] for i in range(12)]                # on and off five times a second
    with pytest.raises(TR.TransitionError, match="at most 3 big flashes a second"):
        plan({"kind": "invert", "from": 1.0, "to": 2.5, "in": 0.0, "out": 0.0, "strength": keys})


def test_turning_the_hues_back_keeps_the_colours_and_swaps_light_and_dark():
    hue = lambda x: float(cv2.cvtColor(x[:1, :1], cv2.COLOR_RGB2HSV)[0, 0, 0])   # noqa: E731
    near = lambda a, b: abs((a - b + 180.0) % 360.0 - 180.0)                    # noqa: E731
    rose = np.full((2, 2, 3), PAL.srgb(DAWN["rose"]), np.float32)
    assert near(hue(ERF.invert(rose, 180.0)), hue(rose)) < 2.0 < 150.0 < near(hue(ERF.invert(rose)), hue(rose))
    cream = np.full((2, 2, 3), PAL.srgb(DAWN["base"]), np.float32)
    assert ERF.luma(ERF.invert(cream, 180.0)).max() < 0.1


# ================================================================================================= digicam
def test_a_digicam_stamps_the_date_in_orange_in_its_corner():
    size = (640, 360)
    pl = plan({"kind": "digicam", "from": 1.0, "to": 3.0, "noise": 0.0, "fringe": 0.0, "barrel": 0.0, "wb": 0.0,
               "hunt": 0.0}, size=size)
    out = runner(pl, size).frame(frame(2.0), np.full((360, 640, 3), 0.25, np.float32))
    orange = (out[..., 0] > 0.8) & (out[..., 1] > 0.3) & (out[..., 1] < 0.65) & (out[..., 2] < 0.3)
    ys, xs = np.nonzero(orange)
    assert orange.sum() > 30 and xs.min() > 640 * 0.5 and ys.min() > 360 * 0.7


def test_a_digicam_is_the_same_frame_twice_and_noisier_than_the_picture():
    size = (160, 90)
    pl = plan({"kind": "digicam", "from": 1.0, "to": 3.0, "date": False}, size=size)
    img = np.full((90, 160, 3), 0.4, np.float32)
    one, two = runner(pl, size).frame(frame(2.0), img), runner(pl, size).frame(frame(2.0), img)
    assert np.array_equal(one, two) and one.std() > 0.01


def test_jpeg_cuts_a_smooth_picture_into_eight_pixel_blocks():
    ramp = np.tile(np.linspace(0.0, 1.0, 128, dtype=np.float32)[None, :, None], (64, 1, 3))
    ramp = ramp + 0.15 * np.sin(np.arange(64, dtype=np.float32) / 5.0)[:, None, None]
    d = np.abs(np.diff(ERF.jpeg(np.clip(ramp, 0.0, 1.0), 8)[..., 1], axis=1)).mean(0)
    edges = d[7::8].mean()
    inside = np.delete(d, np.s_[7::8]).mean()
    assert edges > 1.5 * inside


def test_white_balance_leans_cool_or_warm_and_hunts():
    cool, warm = entry("digicam", wb=-0.8, hunt=0.0), entry("digicam", wb=0.8, hunt=0.0)
    r, _, b = ER.wb_gains(cool, 2.0)
    assert b > 1.0 > r
    r, _, b = ER.wb_gains(warm, 2.0)
    assert r > 1.0 > b
    assert ER.wb_gains(cool, 1.0) == ER.wb_gains(cool, 2.5)
    hunting = entry("digicam", hunt=0.8)
    assert len({ER.wb_gains(hunting, t) for t in (1.0, 1.7, 2.4)}) == 3


def test_seven_segment_glyphs():
    polys, w = ER.glyphs("'05 10 28")
    assert len(polys) == 1 + 6 + 5 + 2 + 6 + 5 + 7 and w > 4.0                 # the tick, 0, 5, 1, 0, 2, 8
    assert ER.glyphs("8")[1] == pytest.approx(ER.SEG_W)
    with pytest.raises(ValueError, match="no seven-segment glyph for 'A'"):
        ER.glyphs("A")


# ================================================================================================= chrome
def test_chrome_on_the_figure_leaves_the_rest_as_it_was():
    size = (160, 90)
    cov = np.zeros((90, 160), np.float32)
    cov[20:80, 60:100] = 1.0
    pl = plan({"kind": "chrome", "from": 1.0, "to": 3.0, "mask": "figure", "objects": "R*", "bloom": 0, "sparkles": 0,
               "bubbles": 0}, size=size)
    img = np.random.default_rng(1).random((90, 160, 3)).astype(np.float32)
    out = runner(pl, size, figure=cov).frame(frame(2.0), img.copy())
    assert np.array_equal(out[cov == 0], img[cov == 0])
    assert not np.allclose(out[cov == 1], img[cov == 1], atol=0.05)


def test_the_sheen_follows_its_phase_and_sheen_zero_is_the_picture():
    img = np.random.default_rng(5).random((40, 60, 3)).astype(np.float32)
    assert np.allclose(ERF.chrome_sheen(img, sheen=0.0), img, atol=1e-6)
    a, b = ERF.chrome_sheen(img, phase=0.0), ERF.chrome_sheen(img, phase=0.3)
    assert a.dtype == np.float32 and 0.0 <= a.min() and a.max() <= 1.0 and not np.allclose(a, b)


def test_a_hit_swings_the_sheen_and_it_settles():
    fx = entry("chrome", speed=0.0)
    a, d = ER.SWING
    assert ER.sheen_phase(fx, 1.5 + d / 2.0, [1.5]) == pytest.approx(a)
    assert ER.sheen_phase(fx, 1.5 + d + 0.01, [1.5]) == 0.0
    assert entry("chrome", speed=0.5)["speed"] == 0.5 and ER.sheen_phase(entry("chrome", speed=0.5), 2.0, []) == 0.5


def test_sparkles_twinkle_and_a_hit_sets_them_all_off():
    fx = entry("chrome", sparkles=6)
    fx["index"] = 0
    calm, hit = ER.sparkles(fx, 2.0, []), ER.sparkles(fx, 2.05, [2.0])
    assert len(calm) == 6 and len(hit) == 12
    assert all(0.0 <= s["size"] <= 1.0 and 0.0 <= s["x"] <= 1.0 and 0.0 <= s["y"] <= 1.0 for s in calm)
    assert len(ER.sparkles(fx, 2.0 + ER.BURST + 0.01, [2.0])) == 6


def test_bubbles_drift_up():
    fx = entry("chrome", bubbles=4)
    fx["index"] = 0
    now, later = ER.bubbles(fx, 1.2), ER.bubbles(fx, 1.5)
    assert all(b["y"] < a["y"] or b["y"] > a["y"] + 0.5 for a, b in zip(now, later))   # up, or round again


# ================================================================================================= test card
def card(size, **spec):
    return plan({"kind": "testcard", "from": 1.0, "to": 2.0, "labels": [], **spec}, size=size)


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (1080, 1080)])
def test_the_grey_scale_and_gratings_flank_the_circle_inside_the_frame(size):
    lay = ER.card_layout(size, card(size)["fx"][0])
    cx, cy, r = lay["circle"]
    assert len(lay["wedge"]) == len(lay["gratings"]) == 6
    for x0, y0, x1, y1 in lay["wedge"] + lay["gratings"]:
        assert math.hypot(min(max(cx, x0), x1) - cx, min(max(cy, y0), y1) - cy) > r      # none inside the circle
        assert lay["edge"] < x0 < x1 < size[0] - lay["edge"] and lay["edge"] < y0 < y1 < size[1] - lay["edge"]
    if size[0] >= size[1]:
        assert lay["wedge"][0][2] < cx < lay["gratings"][0][0]                # left and right of it
    else:
        assert lay["wedge"][0][3] < cy < lay["gratings"][0][1]                # above and below it


def test_the_circle_shows_the_scene_and_the_card_lies_round_it():
    size = (192, 108)
    img = np.random.default_rng(2).random((108, 192, 3)).astype(np.float32)
    out = runner(card(size), size).frame(frame(1.5), img.copy())
    assert np.allclose(out[54, 96], img[54, 96], atol=1e-5)
    assert not np.allclose(out[:12, :12], img[:12, :12], atol=0.05)


def test_the_circle_frames_its_point():
    size = (192, 108)
    yy, xx = np.mgrid[0:108, 0:192].astype(np.float32)
    img = np.dstack([xx / 192.0, yy / 108.0, np.zeros_like(xx)]).astype(np.float32)
    out = runner(card(size, center=[0.25, 0.4]), size).frame(frame(1.5), img)
    assert np.allclose(out[54, 96, :2], [0.25, 0.4], atol=0.01)


def test_the_card_blows_open_at_its_end():
    fx = card((192, 108), open=0.3)["fx"][0]
    shares = [ER.open_share(fx, t) for t in (1.5, 1.7, 1.8, 1.9, 2.0)]
    assert shares[:2] == [0.0, 0.0] and shares[2] < shares[3] < shares[4] == 1.0


def test_the_first_label_sits_on_the_circles_foot():
    font = system_font()
    size = (960, 540)
    pl = plan({"kind": "testcard", "from": 1.0, "to": 2.0, "font": font, "labels": ["PLEASE STAND BY"]}, size=size)
    fx = pl["fx"][0]
    img = np.full((540, 960, 3), 0.5, np.float32)
    out = runner(pl, size, font_of=lambda f: f).frame(frame(1.5), img)
    cx, cy, cap = ER.card_layout(size, fx)["label"]
    box = out[int(cy - cap):int(cy + cap), int(cx - 3 * cap):int(cx + 3 * cap)]
    white = np.asarray(fx["inks"]["white"], np.float32)
    assert np.any(np.all(np.abs(box - white) < 0.05, axis=2))                # its letters
    assert np.any(np.all(np.abs(box - np.asarray(fx["inks"]["black"], np.float32)) < 0.05, axis=2))   # its box


# ================================================================================================= cards
def test_each_hit_sets_off_its_cards_for_their_life():
    fx = plan({"kind": "cards", "from": 1.0, "to": 4.0, "times": [1.5, 2.5], "count": 3, "life": 0.8})["fx"][0]
    starts = ER.card_starts(fx, fx["times"])
    assert len(starts) == 6 and starts[1][0] == pytest.approx(1.5 + ER.CARD_GAP)
    assert ER.cards_at(fx, 1.4, fx["times"], 16 / 9) == []
    flying = ER.cards_at(fx, 1.75, fx["times"], 16 / 9)
    assert len(flying) == 3 and [c[2][2] for c in flying] == sorted((c[2][2] for c in flying), reverse=True)
    assert ER.cards_at(fx, 2.45, fx["times"], 16 / 9) == []


def test_a_card_flies_at_the_lens_or_across_the_frame():
    fx = plan({"kind": "cards", "from": 1.0, "to": 4.0, "times": [1.5]})["fx"][0]
    z = [ER.card_pose(fx, 0, a, 16 / 9)[0][2] for a in (0.0, 0.5, 0.95)]
    assert z[0] > z[1] > z[2]
    across = plan({"kind": "cards", "from": 1.0, "to": 4.0, "times": [1.5], "path": "across"})["fx"][0]
    u = [ER.LENS * p[0] / p[2] for p in (ER.card_pose(across, 0, a, 16 / 9)[0] for a in (0.0, 1.0))]
    assert min(abs(u[0]), abs(u[1])) > 16 / 9 / 2 and u[0] * u[1] < 0       # from beyond one side to the other
    R = ER.card_pose(fx, 0, 0.3, 16 / 9)[1]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_cards_are_drawn_over_the_frame_only_while_they_fly():
    size = (192, 108)
    pl = plan({"kind": "cards", "from": 1.0, "to": 4.0, "times": [1.5]}, size=size)
    img = np.full((108, 192, 3), 0.5, np.float32)
    ffx = runner(pl, size)
    assert np.array_equal(ffx.frame(frame(1.4), img), img)
    changed = np.any(np.abs(ffx.frame(frame(1.9), img) - img) > 1e-3, axis=2)
    assert 0.0 < changed.mean() < 0.9


def test_a_card_shows_its_bars_face_on():
    fx = plan({"kind": "cards", "from": 1.0, "to": 4.0, "times": [1.5], "shadow": 0})["fx"][0]
    faces = ERF.card_faces(fx["palette"], "true")
    flight = [(0, 0.0, np.array([0.0, 0.0, 1.3]), np.eye(3))]                 # in the middle at LENS, square on
    out = ERF.cards(np.zeros((216, 384, 3), np.float32), fx, flight, faces, (1.0, 1.0, 1.0), 1.0)
    tall = ER.LENS * fx["size"] / 1.3 * 216                                   # its height on the frame, px
    assert np.any(out[int(108 - 0.45 * tall)] > 0) and not np.any(out[int(108 - 0.55 * tall)] > 0)   # its top edge
    row = out[int(108 - 0.25 * tall), 192 - 40:192 + 40]                      # across its bars
    assert row.max() > 0.3 and len({tuple(np.round(c, 1)) for c in row}) >= 5
