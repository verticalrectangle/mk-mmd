"""Stacks (mkmmd.core.stack: the keys checked and the copies laid out in time; mkmmd.stack: the picture work) on synthetic
pictures: the plain banded echo drawn as it always was, bands that are wallpapers of sprites, copies in eras' looks, a
tunnel that falls inward forever without a seam, busy clones behind in time that peel off to places of their own, squash
and stretch on the hits about the feet, all of it mirrored and from a shot of its own; and the sprites credited."""
import numpy as np
import pytest
from PIL import Image

from mkmmd import framefx as FF
from mkmmd import mandala as MDF
from mkmmd import stack as ST
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import stack as SK
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 96, 64
RED, BLUE, WHITE, BLACK, GREEN = [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]


def project(fx, **extra):
    return {"project": {"duration": 4.0}, "output": [{"name": "t", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}, {"name": "far", "plate": True}], "fx": list(fx), **extra}


def plan(fx, **extra):
    return TR.plan(project(fx, **extra), FPS, F0, DAWN)


def stack(**kw):
    return {"kind": "stack", "from": 0.5, "to": 3.5, "objects": "Fig*", **kw}


def square(x0, x1, y0, y1):
    m = np.zeros((H, W), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


def blob(cx, cy, rx, ry):
    """A soft-edged ellipse (antialiased over a pixel)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot((xx - cx) / rx, (yy - cy) / ry)
    return np.clip((1.0 - d) * min(rx, ry) + 0.5, 0.0, 1.0).astype(np.float32)


def ramp():
    xx, yy = np.meshgrid(np.linspace(0, 1, W, dtype=np.float32), np.linspace(0, 1, H, dtype=np.float32))
    return np.dstack([xx, yy, np.full_like(xx, 0.3)])


def runner(pl, figure, frames=None, sprite_of=None, plate=None, hits=None, point=None):
    return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: (frames or {}).get(g, ramp()),
                      point_of=lambda key, g: (point, None), figure_of=lambda key, g: figure(g), hits=hits,
                      sprite_of=sprite_of or (lambda refs: []), plate_of=plate)


def frame_at(t):
    return int(round(F0 + t * FPS))


def boxes(img, colour, tol=0.02):
    """(x0, y0, x1, y1) of the pixels of `colour` (x1, y1 past them); None for none."""
    ys, xs = np.nonzero((np.abs(img - np.asarray(colour, np.float32)) < tol).all(-1))
    return None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# ---------------------------------------------------------------- the plain stack, as it always was
def old_stack(img, m, back, copies, step, scale, centre, fill, edge=None, shadow=None, mirror=False):
    """The plain stack as mkmmd.framefx drew it before it had a module of its own: each copy pasted over the whole frame
    farthest first, then the figure; with `mirror` all of it again from its pixels flipped about the centre."""
    def stacked(out, img, m, copies, step, centre, fill, shadow):
        c = (float(centre[0]), float(centre[1]))
        for j in range(len(copies), 0, -1):
            pic, cov, f = copies[j - 1]
            out = FF._paste(out, pic, cov, f, scale ** j, c, (c[0] + j * step[0], c[1] + j * step[1]), edge, shadow)
        return FF._paste(out, img, m, fill, edge=edge, shadow=shadow)

    out = stacked(FF._flat((W, H), back), img, m, copies, step, centre, fill, shadow)
    if not mirror:
        return out
    X = int(round(2.0 * centre[0]))

    def flip(a):
        f = np.zeros_like(a)
        lo, hi = max(X - (W - 1), 0), min(X, W - 1)
        f[:, lo:hi + 1] = a[:, X - hi:X - lo + 1][:, ::-1]
        return f

    def turned(f):
        return {**f, "offset": (-f["offset"][0], f["offset"][1])} if isinstance(f, dict) else f
    sh = None if shadow is None else (shadow[0], (-shadow[1][0], shadow[1][1]), shadow[2], shadow[3])
    return stacked(out, flip(img), flip(m), [(flip(p), flip(c), turned(f)) for p, c, f in copies], (-step[0], step[1]),
                   (X - centre[0], centre[1]), turned(fill), sh)


SILK = {"inks": [[0.1, 0.05, 0.2], [0.9, 0.3, 0.4], [1.0, 0.8, 0.2]], "offset": (2.0, -1.5), "soft": 1.0, "line": 2.0}


@pytest.mark.parametrize("fills, step, scale, centre, mirror", [
    ([RED, BLUE, WHITE], (6.0, -1.5), 1.0, (40.3, 30.7), False),
    ([None, SILK, GREEN], (-7.25, 2.0), 1.0, (52.6, 31.2), True),
    ([SILK, None, RED], (5.5, 0.0), 0.9, (47.5, 30.0), True),
])
def test_a_plain_stack_draws_the_same_pixels_as_before(fills, step, scale, centre, mirror):
    img, m = ramp(), blob(34.3, 33.0, 9.5, 17.0)
    copies = [(img, blob(34.3 + j, 33.0, 9.5, 17.0 - j), f) for j, f in enumerate(fills)]
    edge, shadow = (WHITE, 3.0), (BLACK, (2.5, 3.5), 2.0, 0.5)
    for back in ([0.3, 0.2, 0.4], ramp() * 0.5):
        for fill in (None, BLUE, SILK):
            want = old_stack(img, m, back, copies, step, scale, centre, fill, edge, shadow, mirror)
            got = ST.stack(img, m, back, copies, step, scale, centre, fill, edge, shadow, mirror)
            assert np.allclose(got, want, atol=2e-5)


# ---------------------------------------------------------------- the keys
@pytest.mark.parametrize("own, frag", [
    ({"bands": [{"color": "love"}]}, "`sprites` is required"),
    ({"bands": [{"sprites": "a.png"}]}, "`color` is required"),
    ({"bands": [{"color": "love", "sprites": "a.png", "size": 2}]}, "size = 2"),
    ({"bands": [{"color": "love", "sprites": "a.png", "spin": 1}]}, "unknown keys ['spin']"),
    ({"bands": ["love", ["gold", "pine"], {"color": "love", "sprites": "a.png"}]}, "a band is a colour"),
    ({"bands": [{"color": "love", "sprites": "a.png", "eyes": 3}]}, "eyes = 3"),
    ({"looks": ["halftone", "sepia"]}, "one of none, halftone"),
    ({"recede": {"scale": 1.2}}, "recede scale = 1.2"),
    ({"recede": {"depth": 3}}, "unknown keys ['depth']"),
    ({"peel": {"t": 1.0}}, "`at` is required"),
    ({"peel": {"at": [0.2, 0.9], "ease": "bounce"}}, "peel ease 'bounce'"),
    ({"peel": {"at": [0.2, 0.9]}, "recede": True}, "recede and peel"),
    ({"squash": {"amount": 0.2}}, "give `times`"),
    ({"squash": {"times": [1.0], "amount": 0.95}}, "squash amount = 0.95"),
    ({"delay": []}, "delay = []"),
    ({"delay": [2, -1]}, "delay = -1"),
])
def test_bad_stack_keys_say_what_is_wrong(own, frag):
    with pytest.raises(TR.TransitionError, match=frag.replace("[", r"\[").replace("]", r"\]")):
        plan([stack(**own)])


def test_a_stack_keeps_its_plain_keys_and_takes_the_new_ones():
    fx = plan([stack(bands=["love", {"color": "gold", "sprites": {"tag": "veggie"}, "eyes": True}, "picture"],
                     looks="vhs", recede=True, delay=[2, 4], squash={"lane": "sub"})])["fx"][0]
    assert fx["bands"][0] == FX._colour("love", DAWN, "band") and fx["bands"][2] is None
    band = fx["bands"][1]
    assert SK.is_sprites(band) and band["sprites"] == [{"tag": ["veggie"]}] and band["face"]["eyes"]
    assert band["color"] == FX._colour("gold", DAWN, "band") and band["size"] == SK.BAND["size"]
    assert fx["looks"] == ["vhs"] and fx["recede"] == SK.RECEDE and fx["delay"] == [2, 4]
    assert fx["squash"]["lane"] == "sub" and fx["squash"]["amount"] == SK.SQUASH["amount"] and fx["peel"] is None
    plain = plan([stack()])["fx"][0]
    assert plain["looks"] is None and plain["recede"] is None and plain["squash"] is None and plain["delay"] == 0


# ---------------------------------------------------------------- eat the rainbow: bands of sprites
def disc_sprite(path, colour, size=24):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    a = np.clip(size / 2.0 - np.hypot(xx - (size - 1) / 2.0, yy - (size - 1) / 2.0) + 0.5, 0.0, 1.0)
    rgba = np.dstack([np.full((size, size), c * 255.0, np.float32) for c in colour] + [a * 255.0]).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
    return str(path)


def test_a_band_of_sprites_fills_its_copy_with_a_tinted_wallpaper_that_scrolls_inside_a_line_of_its_colour(tmp_path):
    sprite = disc_sprite(tmp_path / "pea.png", BLUE)
    band = {"color": "#ff0000", "sprites": sprite, "size": 0.25, "tint": 0.5, "line": 54.0, "scroll": [0.25, 0.0]}
    pl = plan([stack(count=1, step=[0.75, 0.0], bands=[band], back="#000000", fill="#ffffff")])
    run = runner(pl, lambda g: square(4, 40, 4, 60), sprite_of=lambda refs: [sprite])
    a, b = run.frame(frame_at(1.0), ramp()), run.frame(frame_at(2.0), ramp())
    copy = (slice(4, 60), slice(52, 88))                        # the square laid 48 px (0.75 of 64) to the right
    assert np.allclose(a[:, :52][a[:, :52].max(-1) > 0.02], WHITE, atol=1e-5)    # the figure, and black round it
    assert np.allclose(np.delete(a, np.s_[52:88], 1)[:, 40:], 0.0)               # nothing of the copy outside it
    inner = a[copy][3:-3, 3:-3]                                 # inside its line (3.2 px: 54 px at 1080, at 64)
    tinted = (np.abs(inner - [0.5, 0.0, 0.5]) < 0.03).all(-1).mean()
    ground = (np.abs(inner - RED) < 0.01).all(-1).mean()
    assert tinted > 0.15 and ground > 0.2                       # blue peas pulled half way to red, on red
    rim = a[copy].copy()
    rim[2:-2, 2:-2] = RED
    assert np.allclose(rim, RED, atol=1e-5)                     # its rim: a line of the band's colour
    assert np.allclose(b[copy][3:-3, 19:-3], a[copy][3:-3, 3:-19], atol=1e-5)   # 16 px further right a second on


def test_a_wallpaper_tile_wraps_round_so_tiles_side_by_side_meet_without_a_seam(tmp_path):
    sprite = disc_sprite(tmp_path / "bean.png", GREEN)
    band = {"color": RED, "tint": 0.0}
    tile, placed = ST.wall_tile([sprite], band, 20.0, 1.0, (1, 2))
    h, w = tile.shape[:2]
    big = ST.wallpaper(tile, (0.0, 0.0), (0, 0, 3 * w, 3 * h))
    redraw = np.empty((3 * h, 3 * w, 3), np.float32)
    redraw[:] = RED
    for t, M, n in placed:                                      # each sprite once a tile, on a canvas three tiles wide
        x, y = M[:, :2] @ np.asarray(t.anchor) + M[:, 2]
        if 0 <= x < w and 0 <= y < h:
            for dx in (0, w, 2 * w):
                for dy in (0, h, 2 * h):
                    MDF.lay(redraw, t, M + np.array([[0, 0, dx], [0, 0, dy]]), WHITE)
    assert np.allclose(big[h:2 * h, w:2 * w], redraw[h:2 * h, w:2 * w], atol=1e-4)
    assert len({n for _, _, n in placed}) == ST.COLS * ST.ROWS


# ---------------------------------------------------------------- the looks
def laid(look, shift=(0, 0), fill=RED, s=1.0):
    """A square copy laid on black in `look`, on a canvas roomy enough that what the look reaches stays in it."""
    m = np.zeros((120, 160), np.float32)
    m[40:80, 60:96] = 1.0
    out = np.zeros((120, 160, 3), np.float32)
    M = ST.scaled(s, (78.0, 60.0), (78.0 + shift[0], 60.0 + shift[1]))
    return ST.lay(out, ST.Layer(None, m, fill, M, look=look), px=1.0, t=1.0)


def test_no_look_draws_the_copy_as_it_is():
    assert np.array_equal(laid("none"), laid(None)) and laid(None)[50, 70].tolist() == RED


@pytest.mark.parametrize("look", ["halftone", "flat", "pixel", "chrome"])
def test_a_look_rides_with_its_copy(look):
    a, b = laid(look), laid(look, (7, 3))                       # (a tape's lines and noise are the screen's)
    assert np.allclose(b[3:, 7:], a[:-3, :-7], atol=1e-5)


def test_halftone_prints_dots_of_ink_on_newsprint():
    out = laid("halftone")[42:78, 62:94]
    paper = (np.abs(out - ST.PAPER) < 0.02).all(-1).mean()
    ink = (out[..., 0] > 0.5) & (out[..., 1] < 0.2) & (out[..., 2] < 0.2)
    assert 0.1 < paper < 0.6 and ink.mean() > 0.3                # red dots on cream, both showing
    for y in range(0, 36 - 10, 5):                              # a screen of dots: ink and paper in every cell
        for x in range(0, 32 - 10, 5):
            cell = (np.abs(out[y:y + 10, x:x + 10] - ST.PAPER) < 0.02).all(-1)
            assert 0 < cell.mean() < 1


def test_flat_pushes_colours_to_vivid_flats_in_a_thick_line():
    out = laid("flat")
    inside = out[50:70, 70:86].reshape(-1, 3)
    assert len(np.unique(np.round(inside, 3), axis=0)) == 1     # one flat colour
    assert np.allclose(out[40, 70], ST.INK, atol=1e-5) and np.allclose(out[60, 60], ST.INK, atol=1e-5)  # its rim


def test_vhs_draws_scanlines_and_bleeds_its_colour_to_the_right():
    out = laid("vhs", fill=[0.8, 0.3, 0.3])
    rows = out[44:76, 70:86].mean(axis=(1, 2))
    assert rows.max() - rows.min() > 0.08                       # scanlines
    assert out[60, 96:102, 0].max() > 0.15 and out[60, 52:58].max() < 0.02   # bled right of it, not left


def test_pixel_lays_the_copy_in_blocks_fixed_to_it():
    out = laid("pixel", fill=[0.3, 0.6, 0.9])
    b = 12                                                      # PIXEL at 1080 px, px 1: from the frame's corner
    for y0 in range(0, 120 - b, b):
        for x0 in range(0, 160 - b, b):
            block = out[y0:y0 + b, x0:x0 + b].reshape(-1, 3)
            assert (block == block[0]).all()                    # every block one colour
    assert np.allclose(np.unique(out[..., 0]), [0.0, 1.0 / 3.0])   # four levels a channel: 0.3 -> 1/3


def test_chrome_turns_the_copy_to_shining_metal():
    out = laid("chrome")
    lum = out[42:78, 62:94] @ ST.LUMA
    assert lum.max() > 0.8 and lum.min() < 0.35                 # highlights and a dark horizon
    assert np.abs(out[42:78, 62:94] - RED).mean() > 0.2         # nothing like its flat red any more


# ---------------------------------------------------------------- the tunnel
def tunnel(**kw):
    return plan([stack(recede={"count": 8, "scale": 0.7, "fade": 0.25, **kw.pop("recede", {})}, **kw)])["fx"][0]


def test_a_tunnel_lays_its_copies_deeper_and_deeper_fading_out_at_the_far_end():
    fx = tunnel()
    cps = SK.copies(fx, 0.5, 0, [], FPS, F0)
    assert [c["depth"] for c in cps] == [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]      # the farthest first; 0 is behind her
    assert [c["band"] for c in cps] == [6, 5, 4, 3, 2, 1, 0] and [c["j"] for c in cps] == [7, 6, 5, 4, 3, 2, 1]
    assert [c["alpha"] for c in cps[:2]] == [0.5, 1.0] and all(c["alpha"] == 1.0 for c in cps[1:])


def test_a_tunnel_falls_a_place_on_each_hit_its_copies_keeping_their_colours():
    fx = tunnel(times=[1.0, 2.0], recede={"snap": 0.2})
    before = {c["j"]: c for c in SK.copies(fx, 0.99, 0, fx["times"], FPS, F0)}
    mid = {c["j"]: c for c in SK.copies(fx, 1.1, 0, fx["times"], FPS, F0)}
    after = {c["j"]: c for c in SK.copies(fx, 1.5, 0, fx["times"], FPS, F0)}
    for j in (1, 2, 3):
        assert before[j]["depth"] == j and after[j]["depth"] == pytest.approx(j + 1)   # a place deeper
        assert j < mid[j]["depth"] < j + 1.15                 # on its way (a little past, and back)
        assert before[j]["band"] == mid[j]["band"] == after[j]["band"]               # its colour rides with it
    assert 0 in after and after[0]["depth"] == pytest.approx(1.0)                    # a new one out from behind her


def test_a_tunnel_has_no_seam_and_comes_round_again():
    def at(p, speed=1.5):
        fx = tunnel(recede={"speed": speed})
        return {round(c["depth"], 6): (c["band"] % 3, round(c["alpha"], 6))
                for c in SK.copies(fx, 0.5 + p / speed, 0, [], FPS, F0)}
    a, b = at(0.9999), at(1.0001)                              # across a whole place: copies move on, nothing jumps
    for d, (band, alpha) in a.items():
        if alpha < 1e-3:                                        # the farthest, all but gone
            continue
        near = min(b, key=lambda e: abs(e - d))
        assert abs(near - d) < 1e-3 and b[near][0] == band and abs(b[near][1] - alpha) < 1e-3
    pl = plan([stack(recede={"count": 6, "speed": 1.5}, bands=["love", "gold", "foam"], center=[0.9, 0.2],
                     back="#000000", fill="#ffffff")])
    run = runner(pl, lambda g: blob(30, 34, 10, 20))
    assert np.allclose(run.frame(frame_at(1.0), ramp()), run.frame(frame_at(3.0), ramp()), atol=1e-6)   # 3 places on
    assert not np.allclose(run.frame(frame_at(1.0), ramp()), run.frame(frame_at(1.5), ramp()), atol=1e-3)


def test_a_tunnel_s_copies_age_as_they_fall_and_ask_for_the_frames_they_show():
    fx = tunnel(delay=4, recede={"speed": 1.0})
    t = 1.3                                                     # 0.8 of a place on, on a frame
    cps = SK.copies(fx, t, 0, [], FPS, F0)
    assert [c["delay"] for c in cps] == [round(4 * c["depth"]) for c in cps] and cps[-1]["delay"] == 3
    f = frame_at(t)
    assert SK.sources(fx, f, F0) == [f - round(4 * c["depth"]) for c in reversed(cps)]
    lane = tunnel(delay=2, lane="kick")
    assert SK.sources(lane, f, F0) == [f - d for d in range(1, 17)]                    # no hits file: all it could show
    assert SK.sources(lane, f, F0, {"kick": [0.75, 1.0]}) == [f - d for d in (2, 4, 6, 8, 10, 12, 14)]


def test_a_tunnel_mirrored_about_its_middle_is_its_own_reflection():
    pl = plan([stack(recede={"count": 7, "speed": 0.8}, bands=["love", "gold", "foam"], looks=["none", "pixel"],
                     center=[0.5, 0.3], mirror=True, back="#000000", fill="#ffffff",
                     edge={"color": "#00ff00", "width": 20})])
    out = runner(pl, lambda g: blob(22, 36, 9, 18)).frame(frame_at(1.3), ramp())
    assert np.allclose(out, out[:, ::-1], atol=1e-5)         # centre x = 0.5 * 96 - 0.5: the pixel columns mirror


# ---------------------------------------------------------------- busy clones
def test_each_copy_shows_her_its_own_frames_behind():
    pl = plan([stack(count=2, delay=[3, 9], step=[0.0, 0.0], bands=["#0000ff", "#00ff00"], back="#000000",
                     fill="#ff0000")])
    fx = pl["fx"][0]
    f = frame_at(1.5)
    assert SK.sources(fx, f, F0) == [f - 3, f - 9] and FX.sources(fx, f, F0) == [f - 3, f - 9]
    run = runner(pl, lambda g: square(g - 30, g - 22, 10, 50))  # she walks right a pixel a frame
    out = run.frame(f, ramp())
    x = f - 30
    assert boxes(out, RED)[0] == x and boxes(out, BLUE)[0] == x - 3 and boxes(out, GREEN)[0] == x - 9


def test_peeled_copies_slide_one_after_another_to_their_own_places():
    peel = {"at": [[0.75, 0.90625], [0.25, 0.90625]], "t": 1.0, "dur": 0.5, "gap": 0.25, "scale": 0.5, "ease": "linear"}
    pl = plan([stack(count=2, delay=[6, 12], step=[0.0625, 0.0], bands=["#0000ff", "#00ff00"], back="#000000",
                     fill="#ff0000", peel=peel)])
    fx = pl["fx"][0]
    assert [SK.peeled(fx, 1, t) for t in (0.9, 1.25, 1.5, 3.0)] == [0.0, 0.5, 1.0, 1.0]
    assert SK.peeled(fx, 2, 1.25) == 0.0 and SK.peeled(fx, 2, 1.5) == 0.5     # one after the other
    assert fx["f0"] in FX.sources(fx, frame_at(2.0), F0)       # the pivot is found on the span's first frame
    run = runner(pl, lambda g: square(40, 56, 10, 50))         # her feet (the pivot) at (47.5, 49.5)
    out = run.frame(frame_at(2.0), ramp())
    for colour, at in ((BLUE, (0.75, 0.90625)), (GREEN, (0.25, 0.90625))):
        x0, y0, x1, y1 = boxes(out, colour)
        assert (x1 - x0, y1 - y0) == (8, 20)                    # half her size
        assert ((x0 + x1) / 2.0 - 0.5, y1 - 0.5) == (at[0] * W - 0.5, at[1] * H - 0.5)   # her feet at `at`
    mid = run.frame(frame_at(1.2), ramp())                     # copy 1 0.4 of the way: from (51.5, 49.5) at her size
    assert boxes(mid, BLUE) == (54, 22, 66, 53)                 # to (71.5, 57.5) at half: 0.8 her size


def test_peeled_copies_of_a_mirrored_stack_go_to_mirrored_places():
    pl = plan([stack(count=1, bands=["#0000ff"], back="#000000", fill="#ff0000", mirror=True, center=[0.5, 0.5],
                     peel={"at": [0.2, 0.9], "t": 0.5, "dur": 0.1})])
    out = runner(pl, lambda g: square(30, 40, 10, 50)).frame(frame_at(2.0), ramp())
    left, right = boxes(out[:, :48], BLUE), boxes(out[:, 48:], BLUE)
    assert left[0] + left[2] + right[0] + 48 + right[2] + 48 == 2 * W     # x and 95 - x


# ---------------------------------------------------------------- squash
def test_a_squash_presses_down_springs_up_past_and_settles_keeping_the_area():
    sq = plan([stack(squash={"times": [1.0], "amount": 0.2, "decay": 0.15, "attack": 0.03})])["fx"][0]["squash"]
    assert SK.squash_at(sq, 0.99, [1.0]) == (1.0, 1.0)
    sx, sy = SK.squash_at(sq, 1.03, [1.0])
    assert sy == pytest.approx(0.8) and sx * sy == pytest.approx(1.0)
    ts = np.linspace(1.03, 1.6, 200)
    sys_ = [SK.squash_at(sq, t, [1.0])[1] for t in ts]
    top = int(np.argmax(sys_))
    assert sys_[top] > 1.05 and ts[top] == pytest.approx(1.03 + SK.REBOUND * 0.15, abs=0.03)   # the stretch
    assert SK.squash_at(sq, 3.0, [1.0])[1] == pytest.approx(1.0, abs=1e-4)


def test_the_figure_squashes_about_its_feet_and_a_copy_behind_in_time_does_it_later():
    pl = plan([stack(count=1, delay=6, step=[0.55, 0.0], bands=["#0000ff"], back="#000000", fill="#ffffff",
                     squash={"times": [1.0], "amount": 0.25, "attack": 0.03})])
    run = runner(pl, lambda g: square(10, 30, 10, 50))
    hit = run.frame(frame_at(1.0 + 1.0 / 30.0), ramp())        # 0.033 s on: pressed (nearly all the way)
    x0, y0, x1, y1 = boxes(hit, WHITE, tol=0.1)
    assert y1 == 50 and 50 - y0 < 34 and x1 - x0 > 22         # her feet where they were; shorter and wider
    assert boxes(hit, BLUE) == (46, 10, 65, 50)                 # her copy (35.2 px on), 6 frames behind, not yet
    later = run.frame(frame_at(1.0 + 7.0 / 30.0), ramp())
    x0, y0, x1, y1 = boxes(later, BLUE, tol=0.1)
    assert y1 == 50 and 50 - y0 < 34                            # now it is


def test_a_squashed_figure_on_its_own_picture_leaves_no_trace_where_it_stood():
    m = square(30, 50, 10, 50)
    pic = ramp()
    pic[m > 0] = GREEN                                          # she is in the picture
    pl = plan([stack(count=1, step=[0.5, 0.0], bands=["#0000ff"], back="picture", fill="picture",
                     squash={"times": [1.0], "amount": 0.3})])
    out = runner(pl, lambda g: m).frame(frame_at(1.0 + 1.0 / 30.0), pic)
    green = (np.abs(out - GREEN) < 0.05).all(-1)
    assert green[30:50, 30:50].mean() > 0.5                     # her squashed self
    assert not green[10:20, 30:50].any()                        # and nothing of her above it


# ---------------------------------------------------------------- with a shot of its own
def test_a_rainbow_stack_from_its_own_shot_with_everything_on(tmp_path):
    sprite = disc_sprite(tmp_path / "radish.png", [1.0, 0.2, 0.4])
    seen = []

    def plate(shot, g):
        seen.append(shot)
        return ramp()
    pl = plan([stack(shot="far", at=[0.2, 0.95], height=0.5, mirror=True, center=[0.5, 0.5], count=3, delay=2,
                     bands=[{"color": "love", "sprites": sprite, "eyes": True}, "picture", "silkscreen"],
                     looks=["halftone", "vhs", "chrome"], squash={"times": [1.0]}, back="#000000",
                     edge={"color": "#ffffff", "width": 6}, shadow={"color": "#000000", "alpha": 0.3})])
    run = runner(pl, lambda g: square(40, 56, 10, 50), sprite_of=lambda refs: [sprite], plate=plate)
    out = run.frame(frame_at(1.05), ramp())
    assert set(seen) == {"far"}
    drawn = out.max(-1) > 0.02
    assert drawn[:, :48].any() and drawn[:, 48:].any() and np.isfinite(out).all()
    left, right = np.nonzero(drawn[:, :48].any(0))[0], np.nonzero(drawn[:, 48:].any(0))[0] + 48
    assert left.min() + right.max() == pytest.approx(W - 1, abs=2)   # both sides of the frame, mirrored


# ---------------------------------------------------------------- credits
def test_every_sprite_a_project_shows_is_credited():
    data = {"fx": [{"kind": "mandala", "rings": [{"source": {"sprites": ["clover_a", {"tag": "clover"}]}}]},
                   {"kind": "stack", "bands": ["love", {"color": "gold", "sprites": {"tag": ["veggie", "veg_red"]}}],
                    "aspect": {"9x16": {"bands": [{"color": "pine", "sprites": "sprites/*.png"}]}}},
                   {"kind": "paintapp", "stamps": {"sprites": ["carrot", "clover_a"]}}],
            "insert": [{"kind": "iris", "ring": {"kind": "plate", "sprites": {"tag": "veggie"}}}]}
    assert MDF.sprite_refs(data) == ["clover_a", {"tag": ["clover"]}, {"tag": ["veggie", "veg_red"]}, "sprites/*.png",
                                     "carrot", {"tag": ["veggie"]}]
