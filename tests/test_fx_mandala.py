"""Mandalas and trails (mkmmd.core.mandala: the entries checked, the rings laid out and timed, the grow schedule, the
trail's ramp, the sticker shapes; mkmmd.mandala: sprites found and drawn as stickers, the copies laid, the trails behind
the figure) on synthetic pictures, and the credits a mandala's sprites earn."""
import numpy as np
import pytest
from PIL import Image

from mkmmd import assets as A
from mkmmd import framefx as FF
from mkmmd import mandala as MDF
from mkmmd.cli.assets import project_credits
from mkmmd.core import framefx as FX
from mkmmd.core import mandala as MD
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 64, 36
RED, GREEN, BLUE, WHITE, BLACK = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]


def project(fx=(), size=(W, H)):
    return {"project": {"duration": 4.0}, "output": [{"name": "o", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}


def plan(fx=(), size=(W, H)):
    return TR.plan(project(fx, size), FPS, F0, DAWN)


def mandala(**kw):
    return {"kind": "mandala", "from": 1.0, "to": 3.0, "objects": "Fig*", "rings": [{}], **kw}


def stickers(**kw):
    return {"kind": "mandala", "from": 1.0, "to": 3.0, "rings": [{"source": {"shape": "heart"}}], **kw}


def entry(spec, size=(W, H)):
    return plan([spec], size)["fx"][0]


def square(x0, x1, y0, y1, size=(W, H)):
    m = np.zeros((size[1], size[0]), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


def runner(pl, size=(W, H), figure=None, frames=None, sprite_of=None, hits=None):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: (frames or {}).get(g), point_of=None,
                      figure_of=(lambda key, g: figure(g)) if figure else None, sprite_of=sprite_of, hits=hits)


def disc_sprite(path, colour=(255, 0, 0), size=40, box=60):
    """A PNG `box` px square, clear but for a disc `size` px across in its middle."""
    yy, xx = np.mgrid[0:box, 0:box].astype(np.float32)
    a = np.clip(size / 2.0 - np.hypot(xx - (box - 1) / 2.0, yy - (box - 1) / 2.0) + 0.5, 0.0, 1.0)
    rgba = np.dstack([np.full((box, box), c, np.float32) for c in colour] + [a * 255.0]).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
    return path


# ---------------------------------------------------------------- the entries
@pytest.mark.parametrize("spec, frag", [
    ({"kind": "mandala", "from": 1, "to": 2, "objects": "Fig*"}, "`rings` is required"),
    (mandala(rings=[{"size": 2}]), "unknown keys ['size']"),
    (mandala(rings=[{"source": "leaf"}]), "source = 'leaf'"),
    (mandala(rings=[{"source": {"shape": "tulip"}}]), "shape = 'tulip': one of clover"),
    (mandala(rings=[{"source": {"shape": "star", "size": 2}}]), "unknown keys ['size'] of a shape"),
    (mandala(rings=[{"source": {"sprites": [3]}}]), "a slug, a file, a glob or {tag"),
    (mandala(rings=[{"source": {"sprites": {"tags": "x"}}}]), "a slug, a file, a glob or {tag"),
    (mandala(rings=[{"source": {"sprites": "a", "shape": "star"}}]), "\"figure\", {sprites"),
    ({"kind": "mandala", "from": 1, "to": 2, "rings": [{"count": 4}]}, "needs the figure's `objects`"),
    ({"kind": "mandala", "from": 1, "to": 2, "center_figure": True, "rings": [{"source": {"shape": "star"}}]},
     "needs the figure's `objects`"),
    (stickers(objects="Fig*"), "`objects` is only the figure"),
    (mandala(rings=[{"source": {"shape": "star"}, "fill": "love"}]), "`fill` draws the figure"),
    (mandala(rings=[{"orient": "outward"}]), "none of radial, upright"),
    (mandala(rings=[{"count": 0}]), "count = 0: expected 1..64"),
    (mandala(rings=[{"mirror": "yes"}]), "mirror is true or false"),
    (mandala(rings=[{}] * 13), "at most 12"),
    (mandala(bloom=0.2), "bloom fires on the mandala's hits"),
    (mandala(times=[1.5], bloom={"amount": 0.2, "spread": 1}), "bloom: unknown keys ['spread']"),
    (mandala(grow={"dur": 0.3}), "give `times`"),
    (mandala(grow={"times": [1.2], "order": "spiral"}), "none of rings, inward, around, random"),
    (mandala(grow={"times": [1.2], "min_db": -3}), "pick from a lane"),
    (mandala(back=[]), "or a list of them"),
    ({"kind": "trails", "from": 1, "to": 2}, "`objects` is required"),
    ({"kind": "trails", "from": 1, "to": 2, "objects": "Fig*", "blend": "multiply"}, "none of over, screen, add"),
    ({"kind": "trails", "from": 1, "to": 2, "objects": "Fig*", "tint": 2}, "tint = 2: expected 0..1"),
    ({"kind": "trails", "from": 1, "to": 2, "objects": "Fig*", "delay": 0}, "delay = 0: expected 1..60"),
    ({"kind": "trails", "from": 1, "to": 2, "objects": "Fig*", "colors": []}, "or a list of them"),
])
def test_bad_mandalas_and_trails_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError, match=frag.replace("[", r"\[").replace("]", r"\]").replace("{", r"\{")):
        plan([spec])


def test_a_mandala_reads_its_figure_only_when_a_ring_or_its_middle_shows_it():
    figs = {f: [it for it in items if it["kind"] == "figure"] for f, items in TR.demands(plan([stickers()])).items()}
    assert not any(figs.values())                                   # stickers alone: no figure layer to render
    pl = plan([mandala(), stickers(center_figure=True, objects="Fig*", **{"from": 3.0, "to": 3.2})])
    fig, mid = pl["fx"]
    assert FX.uses_figure(fig) and FX.uses_figure(mid)
    assert FX.sources(fig, fig["f0"] + 5, F0) == [fig["f0"]]       # copies of her are sized by her first frame
    assert FX.sources(fig, fig["f0"], F0) == [] and FX.sources(mid, mid["f0"] + 3, F0) == []


# ---------------------------------------------------------------- the rings
def three_rings(**kw):
    return entry(mandala(rings=[{"count": 4, "radius": 0.1, "turn": 10.0, "mirror": True},
                                {"count": 6, "radius": 0.2, "turn": 10.0},
                                {"count": 8, "radius": 0.3, "turn": 10.0, "orient": "upright", "spin": 5.0}], **kw))


def test_rings_hold_their_counts_on_their_radii_the_outer_drawn_first_and_the_first_copy_straight_up():
    fx = three_rings()
    cs = MD.copies(fx, 1.0, [], FPS, F0)
    assert [c["ring"] for c in cs] == [2] * 8 + [1] * 6 + [0] * 4
    for c in cs:
        assert np.hypot(c["x"], c["y"]) == pytest.approx(fx["rings"][c["ring"]]["radius"])
    firsts = [c for c in cs if c["j"] == 0]
    assert all(c["x"] == pytest.approx(0.0, abs=1e-12) and c["y"] < 0 for c in firsts)
    ring1 = [c for c in cs if c["ring"] == 1]
    assert [round(c["angle"], 6) for c in ring1] == [60.0 * j for j in range(6)]   # radial: turned as it stands
    assert ring1[1]["x"] < 0 and ring1[1]["y"] < 0                                  # counter-clockwise: up and left


def test_rings_turn_alternate_ways_unless_told_and_upright_copies_spin_on_their_own():
    fx = three_rings()
    at = {c["ring"]: c for c in MD.copies(fx, 2.0, [], FPS, F0) if c["j"] == 0}     # 1 s in: 10 degrees each
    assert np.degrees(np.arctan2(-at[0]["x"], -at[0]["y"])) == pytest.approx(10.0)
    assert np.degrees(np.arctan2(-at[1]["x"], -at[1]["y"])) == pytest.approx(-10.0)
    assert np.degrees(np.arctan2(-at[2]["x"], -at[2]["y"])) == pytest.approx(10.0)
    assert at[2]["angle"] == pytest.approx(5.0)                                     # upright, spinning 5 a second
    same = {c["ring"]: c for c in MD.copies(three_rings(alternate=False), 2.0, [], FPS, F0) if c["j"] == 0}
    assert np.degrees(np.arctan2(-same[1]["x"], -same[1]["y"])) == pytest.approx(10.0)


def test_a_mirrored_ring_flips_every_other_copy_and_drawing_shows_it():
    fx = three_rings()
    assert [c["mirror"] for c in MD.copies(fx, 1.0, [], FPS, F0) if c["ring"] == 0] == [False, True, False, True]
    assert not any(c["mirror"] for c in MD.copies(fx, 1.0, [], FPS, F0) if c["ring"] != 0)
    t = MDF.Tile(np.zeros((5, 9, 3), np.float32), np.ones((5, 9), np.float32), None, (2.0, 2.0))
    flip = MDF.placed(t, (30.0, 20.0), 0.0, 1.0, mirror=True)
    assert flip @ [6.0, 2.0, 1.0] == pytest.approx([26.0, 20.0])                    # 4 px right of the anchor: left


def test_a_ring_steps_on_each_hit_after_its_first_frame_easing_past_and_settling():
    fx = entry(mandala(times=[1.0, 2.0], rings=[{"turn": 0.0, "step": 30.0}, {"turn": 0.0, "step": 30.0}]))
    turns = [MD.turn(fx, 0, t, fx["times"], FPS, F0) for t in (1.9, 2.0, 2.05, 2.1, 2.3, 2.9)]
    assert turns[0] == 0.0 and turns[1] == 0.0                     # the hit on the first frame is where it starts
    assert turns[3] > 30.0 and turns[4] == pytest.approx(30.0) and turns[5] == pytest.approx(30.0)
    assert MD.turn(fx, 1, 2.5, fx["times"], FPS, F0) == pytest.approx(-30.0)        # the next ring the other way
    snapped = entry(mandala(times=[2.0], snap=0.0, rings=[{"turn": 0.0, "step": 30.0}]))
    assert MD.turn(snapped, 0, 2.0, snapped["times"], FPS, F0) == pytest.approx(30.0)


def test_a_bloom_throws_the_rings_out_on_a_hit_the_inner_first_and_settles():
    fx = entry(mandala(times=[1.5], bloom={"amount": 0.2, "grow": 0.1, "lag": 0.05},
                       rings=[{"radius": 0.1}, {"radius": 0.2}]))
    assert MD.bloom(fx, 0, 1.4, fx["times"]) == (1.0, 1.0)
    r0, s0 = MD.bloom(fx, 0, 1.55, fx["times"])
    assert r0 == pytest.approx(1.2) and s0 == pytest.approx(1.1)    # at its peak `attack` (0.05 s) after the hit
    assert MD.bloom(fx, 1, 1.55, fx["times"])[0] < r0              # the outer ring follows `lag` later
    assert MD.bloom(fx, 0, 4.0, fx["times"])[0] == pytest.approx(1.0, abs=1e-3)
    far = [c for c in MD.copies(fx, 1.55, fx["times"], FPS, F0) if c["ring"] == 0][0]
    assert np.hypot(far["x"], far["y"]) == pytest.approx(0.12)


# ---------------------------------------------------------------- the grow
def test_a_grow_pops_the_copies_one_by_one_in_its_order_from_its_times():
    rings = [{"count": 3, "radius": 0.3, "source": {"shape": "star"}}, {"count": 2, "radius": 0.1, "source": {"shape": "star"}}]
    fx = entry(stickers(rings=rings, grow={"times": [1.5], "gap": 0.1}))
    sched = MD.grow_schedule(fx, fx["grow"]["times"])
    assert sorted(sched, key=sched.get) == [(1, 0), (1, 1), (0, 0), (0, 1), (0, 2)]  # the inner ring first, round it
    assert [round(sched[c], 6) for c in sorted(sched, key=sched.get)] == [1.5, 1.6, 1.7, 1.8, 1.9]
    inward = MD.grow_order(entry(stickers(rings=rings, grow={"times": [1.5], "order": "inward"})))
    assert inward[:3] == [(0, 0), (0, 1), (0, 2)]
    around = MD.grow_order(entry(stickers(rings=rings, grow={"times": [1.5], "order": "around"})))
    assert around == [(1, 0), (0, 0), (0, 1), (1, 1), (0, 2)]       # by angle at rest: 0, 0, 120, 180, 240
    two = entry(stickers(rings=rings, grow={"times": [1.5, 2.0], "gap": 0.05}))
    s2 = MD.grow_schedule(two, two["grow"]["times"])
    assert [round(s2[c], 6) for c in [(1, 0), (1, 1), (0, 0), (0, 1), (0, 2)]] == [1.5, 1.55, 1.6, 2.0, 2.05]
    rnd = entry(stickers(rings=rings, grow={"times": [1.5], "order": "random"}))
    assert sorted(MD.grow_order(rnd)) == sorted(sched) and MD.grow_order(rnd) == MD.grow_order(dict(rnd))


def test_a_copy_is_hidden_before_it_pops_overshoots_and_settles_on_its_size():
    fx = entry(stickers(rings=[{"count": 2, "source": {"shape": "star"}}], grow={"times": [1.5], "gap": 0.5, "dur": 0.4}))
    sched = MD.grow_schedule(fx, fx["grow"]["times"])

    def pops(t):
        return [c["pop"] for c in MD.copies(fx, t, [], FPS, F0, sched)]
    assert pops(1.4) == [0.0, 0.0]
    assert pops(1.7)[0] > 1.05 and pops(1.7)[1] == 0.0             # the first past its size half way, the second not yet
    assert pops(3.0) == pytest.approx([1.0, 1.0], abs=1e-3)
    pl = plan([stickers(rings=[{"count": 2, "radius": 0.2, "scale": 0.3, "source": {"shape": "star"}}],
                        grow={"lane": "kick"}, back="#000000")], size=(96, 96))
    run = runner(pl, size=(96, 96), hits={"kick": [2.5]})
    assert run.frame(F0 + 60, np.zeros((96, 96, 3), np.float32)).max() == 0.0      # nothing yet before the lane's kick
    assert run.frame(F0 + 80, np.zeros((96, 96, 3), np.float32)).max() > 0.5


# ---------------------------------------------------------------- the stickers
def test_sticker_shapes_fill_a_unit_box_the_butterfly_is_symmetric_and_the_clover_has_four_leaves():
    for name in MD.SHAPES:
        lo, hi = MD.shape_box(MD.shape_parts(name))
        assert max(hi - lo) == pytest.approx(1.0)
    wings = [p for role, kind, p, _ in MD.shape_parts("butterfly") if kind == "fill"]
    pts = np.concatenate(wings)
    assert np.sort(pts[:, 0]) == pytest.approx(np.sort(-pts[:, 0]))               # mirror images across x = 0
    assert sum(1 for role, kind, _, _ in MD.shape_parts("clover") if role == "main") == 4
    with pytest.raises(ValueError, match="no shape 'tulip'"):
        MD.shape_parts("tulip")


def test_a_shape_ring_reads_its_colours_at_the_hits_plus_the_copy_and_its_shapes_round_the_ring():
    fx = entry(stickers(rings=[{"source": {"shape": ["clover", "heart"], "colors": ["love", "gold", "pine"]}}]))
    src = fx["rings"][0]["source"]
    assert [MD.shape_of(src, j) for j in range(4)] == ["clover", "heart", "clover", "heart"]
    love, gold, pine = (pytest.approx(PAL.srgb(DAWN[s]), abs=1e-5) for s in ("love", "gold", "pine"))
    assert MD.shape_colours(src, 0, 0)[0] == love and MD.shape_colours(src, 0, 1)[0] == gold   # k + j round the list
    own = entry(stickers(rings=[{"source": {"shape": "clover"}}]))["rings"][0]["source"]
    assert MD.shape_colours(own, 0, 0)[0] == pine                  # a clover's own colours by default


def sprite_mandala(tmp_path, edge=True):
    """A disc sprite alone in the middle of a 270 px frame, 40 px across, with a white edge 40 px at 1080 (10 here)."""
    disc_sprite(tmp_path / "s.png")
    spec = {"kind": "mandala", "from": 1.0, "to": 3.0, "back": "#000000",
            "rings": [{"count": 1, "radius": 0.0, "scale": 40 / 270, "source": {"sprites": str(tmp_path / "s.png")}}]}
    if edge:
        spec["edge"] = {"color": "#ffffff", "width": 40}
    return plan([spec], size=(270, 270))


def test_a_sprite_wears_a_die_cut_edge_grown_from_its_alpha_not_its_box(tmp_path):
    run = runner(sprite_mandala(tmp_path), size=(270, 270), sprite_of=MDF.sprite_source())
    out = run.frame(F0 + 45, np.zeros((270, 270, 3), np.float32))
    c = 134.5                                                       # the frame's middle: the sprite's
    assert out[134, 134] == pytest.approx(RED, abs=0.02)
    assert out[134, int(c + 20 + 5)] == pytest.approx(WHITE, abs=0.02)            # its edge: 10 px (40 at 1080) round it
    assert out[134, int(c + 20 + 16)] == pytest.approx(BLACK, abs=0.02)           # past the edge: the back
    d = int(c + 25)                                                 # the PNG's corner, beyond the disc and its edge
    assert out[d, d] == pytest.approx(BLACK, abs=0.02)
    bare = runner(sprite_mandala(tmp_path, edge=False), size=(270, 270), sprite_of=MDF.sprite_source())
    assert bare.frame(F0 + 45, np.zeros((270, 270, 3), np.float32))[134, int(c + 25)] == pytest.approx(BLACK, abs=0.02)


def test_sprites_are_dealt_round_a_ring_by_seed_every_one_before_any_comes_again(tmp_path):
    fx = entry(stickers(rings=[{"count": 12, "source": {"shape": "star"}}, {"source": {"shape": "star"}}]))
    picks = [MD.sprite_pick(fx, 0, j, 4) for j in range(12)]
    assert sorted(picks[:4]) == [0, 1, 2, 3] and picks[4:8] == picks[:4]          # dealt round, then round again
    assert picks == [MD.sprite_pick(dict(fx), 0, j, 4) for j in range(12)]        # the same seed, the same
    other = [MD.sprite_pick(dict(fx, seed=2), 0, j, 8) for j in range(8)]
    assert other != [MD.sprite_pick(fx, 0, j, 8) for j in range(8)]
    assert [MD.sprite_pick(fx, 1, j, 8) for j in range(8)] != [MD.sprite_pick(fx, 0, j, 8) for j in range(8)]
    for i, rgb in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        disc_sprite(tmp_path / f"s{i}.png", rgb)
    spec = {"kind": "mandala", "from": 1.0, "to": 3.0, "back": "#000000",
            "rings": [{"count": 6, "radius": 0.3, "scale": 0.15, "source": {"sprites": "s*.png"}, "turn": 0.0}]}

    def draw(seed):
        pl = plan([dict(spec, seed=seed)], size=(160, 160))
        return runner(pl, size=(160, 160), sprite_of=MDF.sprite_source(tmp_path)).frame(
            F0 + 40, np.zeros((160, 160, 3), np.float32))
    a, b = draw(1), draw(1)
    assert np.array_equal(a, b) and not np.array_equal(a, draw(5))
    colours = {tuple(np.round(a[int(79.5 - 48 * np.cos(np.radians(60 * j))), int(79.5 - 48 * np.sin(np.radians(60 * j)))])
                     ) for j in range(6)}
    assert colours == {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}          # all three round the ring


def test_sprites_are_found_by_slug_tag_file_and_glob_and_what_finds_nothing_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    reg = A.Registry()
    for name, tags in (("b1", ["sprite", "butterfly"]), ("b2", ["sprite", "butterfly"]), ("c1", ["sprite", "clover"])):
        disc_sprite(tmp_path / f"{name}.png")
        reg.add({"kind": "texture", "slug": name, "path": str(tmp_path / f"{name}.png"), "tags": tags})
    reg.add({"kind": "font", "slug": "f", "path": str(tmp_path / "f.ttf"), "tags": ["sprite", "butterfly"]})
    reg.save()
    get = MDF.sprite_source(tmp_path)
    assert [p.split("/")[-1] for p in get([{"tag": ["butterfly"]}])] == ["b1.png", "b2.png"]   # pictures only, by slug
    assert [p.split("/")[-1] for p in get(["c1", "b*.png", "c1.png"])] == ["c1.png", "b1.png", "b2.png"]
    for refs, frag in (([{"tag": ["moth"]}], "has the tags"), (["nope"], "'nope': not a picture"),
                       (["x*.png"], "no pictures"), (["gone.png"], "is not a file")):
        with pytest.raises(FX.FxError, match=frag):
            get(refs)
    pl = plan([stickers(rings=[{"source": {"sprites": {"tag": "moth"}}}])])
    with pytest.raises(FX.FxError, match="has the tags"):                           # before any frame is drawn
        runner(pl, sprite_of=get)


# ---------------------------------------------------------------- the mandala drawn
def test_a_mandala_lays_copies_of_the_figure_round_its_centre_sized_by_her_first_frame_over_its_back():
    fig = square(28, 36, 10, 26, (96, 96))                          # 8 x 16 px, centroid (31.5, 17.5)
    pl = plan([{"kind": "mandala", "from": 1.0, "to": 3.0, "objects": "Fig*", "back": "#0000FF", "fill": "#FF0000",
                "rings": [{"count": 4, "radius": 0.3, "scale": 16 / 96, "turn": 0.0}]}], size=(96, 96))
    out = runner(pl, size=(96, 96), figure=lambda g: fig).frame(F0 + 40, np.zeros((96, 96, 3), np.float32))
    assert out[0, 0].tolist() == BLUE
    for x, y in ((47.5, 18.7), (18.7, 47.5), (47.5, 76.3), (76.3, 47.5)):        # up, left, down, right of the middle
        assert out[int(round(y)), int(round(x))] == pytest.approx(RED, abs=0.02)
    assert out[int(47.5 - 28.8), int(47.5 - 4.5)].tolist() == BLUE               # up: 8 wide, so 4 px each side
    assert out[int(47.5 - 28.8 - 6), int(47.5)] == pytest.approx(RED, abs=0.02)   # and 16 tall, along the radius
    big = square(20, 44, 0, 48, (96, 96))                           # three times taller on the span's first frame
    small = runner(pl, size=(96, 96), figure=lambda g: big if g == F0 + 30 else fig).frame(
        F0 + 40, np.zeros((96, 96, 3), np.float32))
    red = lambda o: int(((o[..., 0] > 0.5) & (o[..., 2] < 0.5)).sum())                # noqa: E731
    assert 0 < red(small) < red(out) / 4                            # its copies a third as tall


def test_a_mandala_s_back_steps_between_the_picture_colours_and_patterns_on_its_hits():
    img = np.full((48, 48, 3), 0.25, np.float32)
    pl = plan([stickers(times=[1.5, 2.0], back=["picture", "#0000FF", {"pattern": "rings", "colors": ["love", "base"]}],
                        rings=[{"count": 1, "radius": 0.0, "scale": 0.05, "source": {"shape": "star"}}])], size=(48, 48))
    run = runner(pl, size=(48, 48))
    assert run.frame(F0 + 40, img)[2, 2].tolist() == [0.25] * 3       # the picture until the first hit
    assert run.frame(F0 + 50, img)[2, 2].tolist() == BLUE
    rings = run.frame(F0 + 70, img)
    assert float(np.ptp(rings[..., 0])) > 0.2                           # then a pattern, not a flat colour


def test_a_mandala_over_the_picture_keeps_the_figure_in_its_middle_over_the_rings():
    img = np.zeros((96, 96, 3), np.float32)
    img[..., 1] = 0.5
    fig = square(40, 56, 30, 66, (96, 96))
    img[30:66, 40:56] = WHITE
    pl = plan([{"kind": "mandala", "from": 1.0, "to": 3.0, "objects": "Fig*", "back": "picture", "center_figure": True,
                "rings": [{"count": 6, "radius": 0.12, "scale": 0.1, "source": {"shape": "star", "colors": ["#FF0000"]}}]}],
              size=(96, 96))
    out = runner(pl, size=(96, 96), figure=lambda g: fig).frame(F0 + 40, img)
    assert out[40:56, 44:52] == pytest.approx(np.ones((16, 8, 3)), abs=1e-5)      # she is over the stars, as she is
    assert out[2, 2] == pytest.approx([0.0, 0.5, 0.0])                             # the picture round them
    assert (np.abs(out - img).sum(-1) > 0.3).any()                                 # and the stars where she is not


# ---------------------------------------------------------------- trails
def trail(**kw):
    return {"kind": "trails", "from": 1.0, "to": 3.0, "objects": "Fig*", "count": 3, "delay": 2, "cycle": 0.0,
            "fade": 1.0, "colors": ["#FF0000", "#00FF00", "#0000FF"], **kw}


def walker(g):
    """A figure 6 px wide walking right 2 px a frame."""
    x = 10 + 2 * (g - (F0 + 30))
    return square(x, x + 6, 10, 26)


def test_a_trail_reads_its_figure_back_count_copies_delay_frames_apart():
    pl = plan([trail()])
    fx = pl["fx"][0]
    f = fx["f0"] + 10
    assert FX.sources(fx, f, F0) == [f - 2, f - 4, f - 6]
    figs = {g for g, items in TR.demands(pl).items() if any(it["kind"] == "figure" for it in items)}
    assert set(range(fx["f0"] - 6, fx["f1"])) <= figs
    assert [(j, g) for j, g, _, _ in MD.trail_copies(fx, f, 1.3, [], FPS, F0)] == [(3, f - 6), (2, f - 4), (1, f - 2)]


def test_a_trail_lays_its_copies_oldest_first_in_ramp_colours_behind_the_figure_as_it_is():
    pl = plan([trail()])
    run = runner(pl, figure=walker)
    f = F0 + 40                                                     # the figure at x 30 .. 35, the copies 26, 22, 18
    img = np.full((H, W, 3), 0.5, np.float32)
    img[10:26, 30:36] = WHITE
    out = run.frame(f, img)
    row = out[15]
    assert row[30:36] == pytest.approx(np.array([WHITE] * 6))      # the figure over all of them
    assert row[26:30] == pytest.approx(np.array([RED] * 4))        # the newest, 2 frames back, over the older
    assert row[22:26] == pytest.approx(np.array([GREEN] * 4)) and row[18:22] == pytest.approx(np.array([BLUE] * 4))
    assert row[17].tolist() == [0.5] * 3 and out[5, 25].tolist() == [0.5] * 3


def test_a_trail_s_colours_run_out_along_it_with_its_cycle_and_its_hits_and_fade_with_age():
    f = F0 + 40
    cycled = runner(plan([trail(cycle=3.0)]), figure=walker).frame(f, np.full((H, W, 3), 0.5, np.float32))
    assert cycled[15, 27] == pytest.approx(BLUE, abs=1e-5)          # 1/3 s in at 3 a second: the newest now the last
    hit = runner(plan([trail(times=[1.2], snap=0.0)]), figure=walker).frame(f, np.full((H, W, 3), 0.5, np.float32))
    assert hit[15, 27] == pytest.approx(BLUE) and hit[15, 23] == pytest.approx(RED)
    faded = runner(plan([trail(fade=0.5)]), figure=walker).frame(f, np.zeros((H, W, 3), np.float32))
    assert faded[15, 29] == pytest.approx([0.5, 0.0, 0.0]) and faded[15, 19] == pytest.approx([0.0, 0.0, 0.125])
    added = runner(plan([trail(blend="add")]), figure=walker).frame(f, np.full((H, W, 3), 0.5, np.float32))
    assert added[15, 29] == pytest.approx([1.0, 0.5, 0.5]) and added[15, 27] == pytest.approx([1.0, 1.0, 0.5])
    stepped = runner(plan([trail(step=[0.0, 0.25])]), figure=lambda g: walker(F0 + 40)).frame(f, np.zeros((H, W, 3),
                                                                                                         np.float32))
    assert stepped[30, 31] == pytest.approx(RED) and stepped[15, 31] == pytest.approx(BLACK)   # held still, its copies
    #                                                                                           9 px further down each
    assert MD.ramp([RED, GREEN], 0.25) == pytest.approx([0.75, 0.25, 0.0]) and MD.ramp([RED, GREEN], -1.0) == GREEN


# ---------------------------------------------------------------- credits
def test_credits_count_the_sprites_a_mandala_shows_by_slug_tag_or_registered_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    root = tmp_path / "proj"
    (root / "sprites").mkdir(parents=True)
    reg = A.Registry()
    for name, tags in (("fly1", ["sprite", "butterfly"]), ("fly2", ["sprite", "butterfly"]), ("leaf", ["sprite"])):
        disc_sprite(root / "sprites" / f"{name}.png")
        reg.add({"kind": "texture", "slug": name, "path": str(root / "sprites" / f"{name}.png"), "tags": tags,
                 "author": "a"})
    disc_sprite(root / "sprites" / "mine.png")                     # a file the registry does not know: not credited

    class Proj:
        pass
    Proj.root, Proj.cast = root, [{"asset": "girl"}]
    Proj.data = {"fx": [{"kind": "mandala", "rings": [{"source": {"sprites": [{"tag": "butterfly"}, "leaf"]}},
                                                      {"source": {"shape": "clover"}}],
                         "aspect": {"9x16": {"rings": [{"source": {"sprites": "sprites/*.png"}}]}}},
                        {"kind": "grid", "rings": [{"source": {"sprites": "nope"}}]}],
                 "credits": {"assets": ["font", "leaf"], "lines": ["Song"]}}
    assert project_credits(Proj, reg) == (["girl", "fly1", "fly2", "leaf", "font"], ["Song"])
    Proj.data["fx"][0]["rings"][0]["source"]["sprites"] = ["missing_slug", "sprites/mine.png"]
    assert project_credits(Proj, reg)[0] == ["girl", "fly1", "fly2", "leaf", "font"]
