"""Photographs as prints (mkmmd.core.prints: the entries checked, the places a scatter, a ring and a pin lay out, the deal
on the hits, a print's fall and settle, the cards' layouts; mkmmd.prints: the photographs found, toned and shown whole on
their cards, a stereocard's two halves from the registry, the prints laid over the picture with their shadows, tape and
pins) on synthetic photographs, and the credits the photographs a prints entry shows earn."""
import math

import numpy as np
import pytest
from PIL import Image

from mkmmd import assets as A
from mkmmd import framefx as FF
from mkmmd import mandala as MDF
from mkmmd import prints as PRF
from mkmmd.cli.assets import project_credits
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import prints as PR
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
MOON = PAL.get("rose-pine-moon")
W, H = 320, 180
GROUND = (0.1, 0.1, 0.12)


def project(fx=(), size=(W, H)):
    return {"project": {"duration": 4.0}, "output": [{"name": "o", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}


def plan(fx=(), size=(W, H), palette=DAWN):
    return TR.plan(project(fx, size), FPS, F0, palette)


def prints(**kw):
    return {"kind": "prints", "from": 1.0, "to": 3.0, "photos": ["p*.png"], **kw}


def entry(spec, size=(W, H), palette=DAWN):
    return plan([spec], size, palette)["fx"][0]


def photo(path, colour, size=(40, 30)):
    """A flat photograph `size` (w, h) px of one colour (0..255), as a PNG."""
    Image.fromarray(np.full((size[1], size[0], 3), colour, np.uint8)).save(path)
    return str(path)


def runner(pl, size=(W, H), root=None, hits=None, point=None):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None,
                      point_of=(lambda key, g: (point, None)) if point else None, hits=hits,
                      sprite_of=MDF.sprite_source(root))


def ground(size=(W, H)):
    return np.full((size[1], size[0], 3), GROUND, np.float32)


def frame_of(t):
    return int(round(F0 + t * FPS))


# ---------------------------------------------------------------- the entries
@pytest.mark.parametrize("spec, frag", [
    ({"kind": "prints", "from": 1, "to": 2}, "`photos` is required"),
    (prints(photos=[]), "photos = []: expected slugs"),
    (prints(photos=[{"tags": "x"}]), "photo {'tags': 'x'}: a slug, a file, a glob or {tag"),
    (prints(mode="grid"), "mode 'grid' is none of scatter, ring, pin"),
    (prints(radius=0.3), "`radius` belongs to mode = \"ring\", not \"scatter\""),
    (prints(mode="ring", base={"photo": "x"}), "`base` belongs to mode = \"pin\", not \"ring\""),
    (prints(mode="ring", region=[0, 0, 1, 1]), "`region` is a scatter's or a pin's"),
    (prints(mode="pin"), "mode = \"pin\" needs a `base`"),
    (prints(mode="pin", base={"photo": ""}), "base photo = '': a registry slug or a file"),
    (prints(mode="pin", base={"photo": "b.png", "height": 0.8}), "base = {"),
    (prints(region=[0.5, 0.0, 0.4, 1.0]), "x0 <= x1 and y0 <= y1"),
    (prints(region=[0.0, 0.0, 1.0]), "expected [x0, y0, x1, y1]"),
    (prints(style="cabinet"), "style 'cabinet' is none of print, stereo"),
    (prints(fix="glue"), "fix 'glue' is none of tape, pins, none"),
    (prints(order="random"), "order 'random' is none of given, shuffle"),
    (prints(count=0), "count = 0"),
    (prints(card={"colour": "base"}), "card = {"),
    (prints(tape={"alpha": 2}), "tape alpha = 2"),
    (prints(tone="sepia"), "tone = 'sepia': one of own, albumen, palette"),
    (prints(tone=["base"]), "tone = ['base']: one of own"),
    (prints(min_db=-3), "pick from a lane"),
])
def test_bad_prints_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError, match=frag.replace("[", r"\[").replace("]", r"\]").replace("{", r"\{")):
        plan([spec])


def test_each_mode_has_its_own_defaults_and_the_kind_takes_a_rect():
    assert set(PR.KINDS) <= set(FX.KINDS) and {"rect", "feather", "strength"} <= FX.KEYS["prints"]
    sc, ring = entry(prints()), entry(prints(mode="ring", center={"point": "obj('o').loc"}))
    pin = entry(prints(mode="pin", base={"photo": "b.png"}))
    assert (sc["count"], sc["size"], sc["fix"], sc["region"]) == (6, 0.3, "tape", PR.REGION["scatter"])
    assert (ring["count"], ring["size"], ring["fix"], ring["region"]) == (8, 0.24, "tape", None)
    assert (pin["count"], pin["size"], pin["fix"], pin["region"]) == (5, 0.16, "pins", PR.REGION["pin"])
    assert sc["center"] is None and ring["center"] == {"mode": "point", "expr": "obj('o').loc"}
    assert FX.points(ring) == [("x0", "obj('o').loc")] and FX.points(sc) == []
    assert sc["times"] == [] and sc["tone"] is None and sc["shadow"]["alpha"] == FX.SHADOW["alpha"]
    assert entry(prints(shadow=False))["shadow"] is None
    assert pin["base"] == {"photo": "b.png", "at": [0.5, 0.5], "size": 0.95, "angle": -1.2, "t": None, "tone": None}
    assert entry(prints(photos={"tag": "crew"}))["photos"] == [{"tag": ["crew"]}]


def test_tones_resolve_from_the_palette_and_keep_their_shape_in_both():
    alb = entry(prints(tone="albumen"))["tone"]
    assert len(alb) == len(PR.ALBUMEN) and alb[0] == pytest.approx([0x23 / 255.0, 0x1b / 255.0, 0x1b / 255.0], abs=1e-5)
    for pal in (DAWN, MOON):                                     # dark to light whichever way the palette runs
        dark, mid, light = entry(prints(tone="palette"), palette=pal)["tone"]
        assert PR._luma(dark) < PR._luma(light)
        assert mid == pytest.approx(FX._colour("gold", pal, "t"))
        assert {tuple(dark), tuple(light)} == {tuple(FX._colour("base", pal, "t")), tuple(FX._colour("text", pal, "t"))}
    own = entry(prints(tone=["pine", "#ffffff"]))["tone"]
    assert own[1] == pytest.approx([1.0, 1.0, 1.0])
    pin = entry(prints(mode="pin", tone="albumen", base={"photo": "b.png", "tone": "own"}))
    assert pin["base"]["tone"] is None and len(pin["tone"]) == 9
    assert entry(prints(mode="pin", tone="albumen", base={"photo": "b.png"}))["base"]["tone"] == pin["tone"]


# ---------------------------------------------------------------- the places
def test_a_scatter_spreads_its_places_inside_its_region_kept_a_print_inside():
    fx = entry(prints(count=7, region=[0.1, 0.2, 0.9, 0.8], size=0.2))
    pts = PR.scatter_places(fx, 16 / 9.0)
    assert len(pts) == 7 and pts == PR.scatter_places(fx, 16 / 9.0)        # seeded: the same every time
    for x, y in pts:                                         # short sides: the frame is 16/9 x 1
        assert 0.1 * 16 / 9.0 + 0.1 - 1e-9 <= x <= 0.9 * 16 / 9.0 - 0.1 + 1e-9
        assert 0.2 + 0.1 - 1e-9 <= y <= 0.8 - 0.1 + 1e-9
    d = [math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(pts) for b in pts[i + 1:]]
    rng = np.random.default_rng(0)                           # spread out: farther apart than places drawn at random
    rand = [min(math.hypot(*(p - q)) for i, p in enumerate(r) for q in r[i + 1:])
            for r in (rng.uniform([0.278, 0.3], [1.5, 0.7], (7, 2)) for _ in range(50))]
    assert min(d) > np.median(rand)
    assert pts != PR.scatter_places(entry(prints(count=7, region=[0.1, 0.2, 0.9, 0.8], size=0.2, seed=2)), 16 / 9.0)
    tall = PR.scatter_places(fx, 9 / 16.0)                   # a tall frame: 1 x 16/9 short sides
    assert all(0.2 - 1e-9 <= x <= 0.8 + 1e-9 and 16 / 9.0 * 0.2 + 0.1 - 1e-9 <= y <= 16 / 9.0 * 0.8 - 0.1 + 1e-9
               for x, y in tall)
    small = entry(prints(region=[0.4, 0.4, 0.5, 0.5], size=0.5))          # too small to keep a print inside: its middle
    assert all(p == pytest.approx((0.45 * 16 / 9.0, 0.45)) for p in PR.scatter_places(small, 16 / 9.0))


def test_a_ring_s_places_stand_evenly_round_it_from_straight_up_and_turn_counter_clockwise():
    fx = entry(prints(mode="ring", count=4, radius=0.3, turn=90.0, phase=0.0))
    at = [PR.ring_offset(fx, j, fx["from"]) for j in range(4)]
    assert at[0] == pytest.approx((0.0, -0.3)) and at[1] == pytest.approx((-0.3, 0.0))   # up, then left
    assert at[2] == pytest.approx((0.0, 0.3)) and at[3] == pytest.approx((0.3, 0.0))
    assert PR.ring_offset(fx, 0, fx["from"] + 1.0) == pytest.approx((-0.3, 0.0))         # 90 degrees a second


def test_a_pin_s_places_lie_in_its_region_of_the_base_photograph():
    fx = entry(prints(mode="pin", count=5, region=[0.1, 0.6, 0.9, 0.95], base={"photo": "b.png"}))
    pts = PR.pin_places(fx, 1.25, (0.05, 0.1))
    assert len(pts) == 5
    assert all(0.15 - 1e-9 <= u <= 0.85 + 1e-9 and 0.7 - 1e-9 <= v <= 0.85 + 1e-9 for u, v in pts)


def test_cards_hold_their_photographs_whole_and_tape_crosses_two_corners():
    w, h, (x, y) = PR.card_layout(200, 100, 0.1, 2.0)
    assert (w, h, x, y) == (220, 130, 10, 10)                # border 10 round, the foot twice that
    ch, wins = PR.stereo_layout(400.0, [1.0, 1.25])
    assert ch == 200.0
    (x0, y0, w0, h0), (x1, y1, w1, h1) = wins
    assert h0 == h1 and w0 / h0 == pytest.approx(1.0) and w1 / h1 == pytest.approx(1.25)   # each its own shape
    assert x1 - (x0 + w0) == pytest.approx(12.0)                                             # the gap between
    assert x0 == pytest.approx(400.0 - (x1 + w1)) and x0 >= 0.045 * 400 - 1e-9               # in the middle
    strips = PR.tape_strips(100.0, 80.0, 0, 0.2, 0.5)        # TL and BR
    assert len(strips) == 2
    (cx, cy), angle, sw, sl = strips[0]
    assert (sw, sl) == pytest.approx((16.0, 40.0)) and angle == 45.0 and cx == pytest.approx(cy) and 0 < cx < 20
    assert 2 * cx * math.sqrt(2.0) == pytest.approx(PR.TAPE_ON * sl)    # that much of its length over the card
    (bx, by), angle, _, _ = strips[1]
    assert (100.0 - bx, 80.0 - by) == pytest.approx((cx, cy)) and angle == 45.0


# ---------------------------------------------------------------- the deal
def test_one_print_lands_on_each_hit_and_without_hits_every_place_holds_one_from_the_start():
    fx = entry(prints(times=[2.5, 1.2, 3.4, 1.8]))
    assert PR.landings(fx, fx["times"]) == [1.2, 1.8, 2.5]   # in order, those before the span's end
    assert PR.landings(fx, []) == []                         # a lane with no hits in the span deals nothing
    assert PR.landings(entry(prints(count=3)), None) == [-math.inf] * 3
    lands = [1.2, 1.8, 2.5]
    assert PR.shown(fx, 1.0, lands) == [] and PR.shown(fx, 1.2 - fx["drop"], lands) == [0]
    assert PR.shown(fx, 2.0, lands) == [0, 1]


def test_a_print_falls_faster_and_faster_lands_on_its_hit_hops_and_settles():
    fx = entry(prints(drop=0.2, settle=0.4, times=[2.0]))
    assert PR.fall(fx, 1.79, 2.0) is None                    # not yet falling
    hs = [PR.fall(fx, 1.8 + 0.2 * i / 10, 2.0)[0] for i in range(10)]
    assert hs[0] == pytest.approx(1.0) and all(a > b for a, b in zip(hs, hs[1:]))
    steps = np.diff(hs)
    assert all(a > b for a, b in zip(steps, steps[1:]))      # gathering speed: each step down longer
    h, r = PR.fall(fx, 2.0, 2.0)
    assert h == 0.0 and r == pytest.approx(PR.LAND)          # on the page on its hit, its slide not done
    assert PR.fall(fx, 2.0 - 1e-6, 2.0)[1] == pytest.approx(PR.LAND, abs=1e-4)               # no jump
    hop = max(PR.fall(fx, 2.0 + 0.4 * i / 40, 2.0)[0] for i in range(40))
    assert 0.0 < hop <= PR.BOUNCE + 1e-9
    rs = [PR.fall(fx, 2.0 + 0.4 * i / 40, 2.0)[1] for i in range(41)]
    assert min(rs) < 0.0                                     # its slide eases out past its rest a little
    assert PR.fall(fx, 2.4, 2.0) == (0.0, 0.0) and PR.fall(fx, 9.0, 2.0) == (0.0, 0.0)
    assert PR.fall(fx, 0.0, -math.inf) == (0.0, 0.0)         # laid before the start: at rest
    assert PR.pin_push(1.99, 2.0) is None and PR.pin_push(2.0, 2.0) == pytest.approx(1.0 + PR.PIN_POP)
    assert PR.pin_push(2.0 + PR.PIN_PUSH, 2.0) == pytest.approx(1.0) and PR.pin_push(5.0, -math.inf) == 1.0


def test_photographs_are_dealt_as_given_or_shuffled_every_one_before_any_comes_again():
    fx = entry(prints())
    assert [PR.photo_pick(fx, k, 3) for k in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    sh = entry(prints(order="shuffle"))
    picks = [PR.photo_pick(sh, k, 5) for k in range(15)]
    assert all(sorted(picks[i:i + 5]) == list(range(5)) for i in (0, 5, 10))
    assert picks != [PR.photo_pick(entry(prints(order="shuffle", seed=4)), k, 5) for k in range(15)]
    tilt, spin, (ox, oy), corners = PR.jitter(fx, 3)
    assert abs(tilt) <= fx["tilt"] and 0.5 * fx["spin"] <= abs(spin) <= fx["spin"] and corners in (0, 1, 2)
    assert math.hypot(ox, oy) <= 0.25 * fx["size"] + 1e-9 and PR.jitter(fx, 3) == (tilt, spin, (ox, oy), corners)


# ---------------------------------------------------------------- drawing
def test_a_print_shows_its_photograph_whole_on_a_cream_card_where_it_lands(tmp_path):
    photo(tmp_path / "p1.png", (200, 30, 30), (60, 40))
    pl = plan([prints(count=1, tilt=0.0, spin=0.0, shadow=False, fix="none", region=[0.5, 0.5, 0.5, 0.5],
                      times=[1.5], card={"border": 0.2})])
    run = runner(pl, root=tmp_path)
    fx = run.fxs[0]
    assert np.array_equal(run.frame(frame_of(1.2), ground()), ground())          # before its fall: nothing
    out = run.frame(frame_of(2.5), ground())                                     # at rest
    laid = run.prints_at(fx, frame_of(2.5))
    assert [p.k for p in laid] == [0] and laid[0].h == 0.0
    tile = laid[0].tile
    x0, y0, pw, ph = tile.photo
    assert abs(pw - ph * 60 / 40.0) <= 1.0                                      # its shape kept, nothing cut away
    want = PRF.photo_at(tmp_path / "p1.png", pw, ph)
    a = tile.a[y0:y0 + ph, x0:x0 + pw, None]
    assert tile.rgb[y0:y0 + ph, x0:x0 + pw] / a == pytest.approx(want, abs=1e-5)   # every pixel of it on the card
    assert max(tile.a.shape) - 2 * 2 == pytest.approx(0.3 * H, abs=2)           # the card's longer side: `size`
    cx, cy = W / 2.0 - 0.5, H / 2.0 - 0.5                                       # its card's middle on the region's
    px = lambda dx, dy: out[int(round(cy + dy)), int(round(cx + dx))]          # noqa: E731
    photo_mid = (x0 + pw / 2.0 - 0.5 - tile.anchor[0], y0 + ph / 2.0 - 0.5 - tile.anchor[1])
    assert px(*photo_mid) == pytest.approx(np.array([200, 30, 30]) / 255.0, abs=0.02)
    border = 0.2 * min(pw, ph)
    card = np.asarray(FX._colour(PR.CARD["color"], DAWN, "t"), np.float32)
    assert px(x0 - border / 2.0 - tile.anchor[0], photo_mid[1]) == pytest.approx(card, abs=0.06)   # its left border
    assert np.array_equal(out[2, 2], np.asarray(GROUND, np.float32))            # the picture round it untouched


def test_a_print_falls_larger_onto_the_page_its_shadow_drawing_in_as_it_lands(tmp_path):
    photo(tmp_path / "p1.png", (30, 200, 30))
    pl = plan([prints(count=1, tilt=0.0, region=[0.5, 0.5, 0.5, 0.5], times=[1.5], drop=0.2)])
    run = runner(pl, root=tmp_path)
    fx = run.fxs[0]
    mid, rest = run.prints_at(fx, frame_of(1.4))[0], run.prints_at(fx, frame_of(2.5))[0]
    scale = lambda M: math.hypot(M[0][0], M[1][0])                               # noqa: E731
    assert scale(mid.M) > scale(rest.M) == pytest.approx(1.0) and mid.h > 0.0 == rest.h
    s_mid, s_rest = run._pr_shadow(fx, mid.h), run._pr_shadow(fx, rest.h)
    assert s_mid[1][1] > s_rest[1][1] and s_mid[2] > s_rest[2] and s_mid[3] < s_rest[3]
    frames = [frame_of(1.3), frame_of(1.4), frame_of(1.5), frame_of(1.6), frame_of(2.5)]
    a = {g: run.frame(g, ground()) for g in frames}
    b = {g: runner(pl, root=tmp_path).frame(g, ground()) for g in reversed(frames)}
    assert all(np.array_equal(a[g], b[g]) for g in frames)                      # any frame, any order: the same


def test_strength_mixes_the_prints_in_and_a_rect_keeps_them_in_it(tmp_path):
    photo(tmp_path / "p1.png", (200, 200, 30))
    one = prints(count=1, region=[0.5, 0.5, 0.5, 0.5])
    full = runner(plan([one]), root=tmp_path).frame(frame_of(2.0), ground())
    half = runner(plan([dict(one, strength=0.5)]), root=tmp_path).frame(frame_of(2.0), ground())
    assert half == pytest.approx((full + ground()) / 2.0, abs=1e-5)
    none = runner(plan([dict(one, strength=0.0)]), root=tmp_path).frame(frame_of(2.0), ground())
    assert np.array_equal(none, ground())
    left = runner(plan([dict(one, rect=[0.0, 0.0, 0.5, 1.0])]), root=tmp_path).frame(frame_of(2.0), ground())
    assert np.array_equal(left[:, W // 2 + 1:], ground()[:, W // 2 + 1:])
    assert left[:, :W // 2 - 1] == pytest.approx(full[:, :W // 2 - 1], abs=1e-6) and not np.array_equal(full, ground())


def test_the_albumen_tone_takes_a_grey_photograph_through_hart_s_prints_tones(tmp_path):
    photo(tmp_path / "p1.png", (128, 128, 128), (50, 50))
    pic = PRF.photo_at(tmp_path / "p1.png", 20, 20, FX._colours(PR.ALBUMEN, DAWN, "t"))
    want = np.asarray([FX._colour(c, DAWN, "t") for c in PR.ALBUMEN[4:5]][0])  # mid grey: the middle of nine
    assert pic[10, 10] == pytest.approx(want, abs=0.02)
    assert PRF.photo_at(tmp_path / "p1.png", 20, 20)[10, 10] == pytest.approx([128 / 255.0] * 3, abs=1e-3)


def test_a_stereocard_shows_both_halves_of_its_stereograph_from_the_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    left = photo(tmp_path / "pl.png", (220, 20, 20), (50, 50))
    right = photo(tmp_path / "pr.png", (20, 20, 220), (50, 50))
    lone = photo(tmp_path / "q.png", (20, 220, 20), (50, 50))
    reg = A.Registry()
    reg.add({"kind": "texture", "slug": "pair", "path": left, "extra": {"stereo": {"right": right}}})
    reg.add({"kind": "texture", "slug": "lone", "path": lone})
    reg.save()
    assert PRF.stereo_halves(left) == (left, right) and PRF.stereo_halves(lone) == (lone, lone)   # alone: twice
    tile = PRF.stereo_tile((left, right), 200, entry(prints())["card"])
    x0, y0, w0, h0 = tile.photo
    seen = tile.rgb / np.maximum(tile.a, 1e-6)[..., None]
    row = seen[int(y0 + 0.7 * h0)]
    reds = np.nonzero((row[:, 0] > 0.7) & (row[:, 2] < 0.3))[0]
    blues = np.nonzero((row[:, 2] > 0.7) & (row[:, 0] < 0.3))[0]
    assert reds.size and blues.size and reds.max() < blues.min()                 # the left half left, the right right
    card = np.asarray(FX._colour(PR.CARD["color"], DAWN, "t"), np.float32)
    assert seen[y0 + 1, x0 + 1] == pytest.approx(card, abs=0.08)                # its arched top: card in the corner
    assert seen[y0 + 1, x0 + w0 // 2] == pytest.approx(np.array([220, 20, 20]) / 255.0, abs=0.03)   # the arch's top
    assert tile.a.shape[1] - 2 * 2 == pytest.approx(200, abs=1) and tile.a.shape[0] - 2 * 2 == pytest.approx(100, abs=1)
    run = runner(plan([prints(photos=["pair"], style="stereo", count=1)]), root=tmp_path)
    assert run._pr_halves[left] == (left, right)
    reg.update("pair", extra={"stereo": {"right": str(tmp_path / "gone.png")}})
    reg.save()
    with pytest.raises(FX.FxError, match="the other half of its stereograph"):
        runner(plan([prints(photos=["pair"], style="stereo")]), root=tmp_path)


def test_pins_go_in_as_a_print_lands_and_a_pin_page_lays_its_base_under_the_prints(tmp_path):
    photo(tmp_path / "p1.png", (30, 200, 30), (40, 40))
    photo(tmp_path / "base.png", (250, 250, 250), (160, 100))
    spec = prints(mode="pin", count=2, base={"photo": "base.png", "angle": 0.0}, times=[1.5, 2.0], tilt=0.0)
    run = runner(plan([spec]), root=tmp_path)
    fx = run.fxs[0]
    falling = run.prints_at(fx, frame_of(1.45))
    assert [p.k for p in falling] == [-1, 0] and falling[1].pins is None          # the base first; no pin yet
    landed = run.prints_at(fx, frame_of(1.6))
    assert landed[1].pins is not None and landed[1].pins >= 1.0
    base = landed[0]
    x0, y0, pw, ph = base.tile.photo
    to_frame = lambda M, p: (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2], M[1][0] * p[0] + M[1][1] * p[1] + M[1][2])  # noqa: E731
    bx0, by0 = to_frame(base.M, (x0, y0))
    bx1, by1 = to_frame(base.M, (x0 + pw, y0 + ph))
    at = to_frame(landed[1].M, landed[1].tile.anchor)
    reg = fx["region"]
    assert bx0 + reg[0] * (bx1 - bx0) - 1 <= at[0] <= bx0 + reg[2] * (bx1 - bx0) + 1     # on its region of the base
    assert by0 + reg[1] * (by1 - by0) - 1 <= at[1] <= by0 + reg[3] * (by1 - by0) + 1
    out = run.frame(frame_of(2.6), ground())
    assert out[int(at[1]), int(at[0])] == pytest.approx(np.array([30, 200, 30]) / 255.0, abs=0.03)
    assert out[int(by0 + 0.2 * (by1 - by0)), int((bx0 + bx1) / 2)] == pytest.approx([250 / 255.0] * 3, abs=0.03)


def test_photographs_that_are_not_there_are_errors_before_any_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    with pytest.raises(FX.FxError, match="fx 0 \\(prints\\) photos: "):
        runner(plan([prints(photos=["nope"])]), root=tmp_path)
    photo(tmp_path / "p1.png", (1, 2, 3))
    with pytest.raises(FX.FxError, match="fx 0 \\(prints\\) photos: "):
        runner(plan([prints(mode="pin", base={"photo": "missing.png"})]), root=tmp_path)


def test_a_breathing_palette_redraws_a_print_s_card_in_its_colours(tmp_path):
    photo(tmp_path / "p1.png", (200, 30, 30))
    run = runner(plan([prints(count=1, region=[0.5, 0.5, 0.5, 0.5], card={"color": "love"})]), root=tmp_path)
    fx = dict(run.fxs[0])
    a = run._pr_tile(fx, run._pr_photos[0][0], 100, 0, 0)
    fx["card"] = dict(fx["card"], color=list(FX._colour("pine", DAWN, "t")))
    b = run._pr_tile(fx, run._pr_photos[0][0], 100, 0, 0)
    assert a is not b and not np.array_equal(a.rgb, b.rgb)                     # the tile follows its colours


# ---------------------------------------------------------------- credits
def test_credits_count_the_photographs_prints_show_by_slug_tag_or_registered_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    root = tmp_path / "proj"
    (root / "photos").mkdir(parents=True)
    reg = A.Registry()
    for name, tags in (("crew1", ["photo", "crew"]), ("crew2", ["photo", "crew"]), ("camp", ["photo"]),
                       ("spike", ["photo"])):
        photo(root / "photos" / f"{name}.png", (10, 10, 10))
        reg.add({"kind": "texture", "slug": name, "path": str(root / "photos" / f"{name}.png"), "tags": tags,
                 "author": "a", "license": "public domain"})
    photo(root / "photos" / "mine.png", (1, 1, 1))                             # not in the registry: not credited

    class Proj:
        pass
    Proj.root, Proj.cast = root, [{"asset": "girl"}]
    Proj.data = {"fx": [{"kind": "prints", "photos": [{"tag": "crew"}, "photos/mine.png"],
                         "aspect": {"9x16": {"photos": "photos/camp.png"}}},
                        {"kind": "prints", "mode": "pin", "photos": "crew1", "base": {"photo": "spike"}},
                        {"kind": "mandala", "photos": ["camp"]}],              # another kind's `photos`: not prints'
                 "credits": {"assets": ["font"], "lines": ["Song"]}}
    assert PRF.photo_refs(Proj.data) == [{"tag": ["crew"]}, "photos/mine.png", "photos/camp.png", "crew1", "spike"]
    assert project_credits(Proj, reg) == (["girl", "crew1", "crew2", "camp", "spike", "font"], ["Song"])
    Proj.data["fx"][0]["photos"] = ["missing_slug"]
    assert project_credits(Proj, reg)[0] == ["girl", "camp", "crew1", "spike", "font"]
