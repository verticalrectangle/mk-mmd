"""The remapping frame effects ([[fx]] kinds "kaleido", "tunnel" and "melt": mkmmd.core.remap checks, times and maps,
mkmmd.remap draws): their keys and errors, the layers they ask for, the kaleidoscope's wedges (one wedge the picture as
it is, its copies turned and mirrored about the right lines, the source turning and stepping on hits), the tunnel's
self-similarity (a level down is the same picture, a level fallen through is where it started), the melts' fields
(bounded, nothing at amount 0, smooth in time, seeded), the figure kept or remapped alone under a mask, and the
strength."""
import math

import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import remap as RMF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import remap as RM
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 96, 64


def plan(*fx, duration=2.0, size=(W, H)):
    data = {"project": {"duration": duration}, "output": [{"name": "t", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": duration}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


def entry(kind, **kw):
    return {"kind": kind, "from": 0.0, "to": 1.0, **kw}


def runner(pl, figure=None, size=(W, H), output=None):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, output=output,
                      figure_of=None if figure is None else (lambda key, g: figure))


def texture(seed=1, size=(W, H)):
    """Smooth colour noise: where a pixel came from can be told by its colour."""
    import cv2
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((size[1], size[0], 3)).astype(np.float32), (0, 0), 1.5)
    return ((t - t.min()) / (t.max() - t.min())).astype(np.float32)


def ramp(size=(W, H)):
    """Red is the column and green the row (0..1): a pixel's colour says where it was read."""
    xx, yy = np.meshgrid(np.linspace(0, 1, size[0], dtype=np.float32), np.linspace(0, 1, size[1], dtype=np.float32))
    return np.dstack([xx, yy, np.zeros_like(xx)])


def disc(cx, cy, r, size=(W, H)):
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    return np.clip(r + 0.5 - np.hypot(xx - cx, yy - cy), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- the entries
@pytest.mark.parametrize("spec, frag", [
    (entry("kaleido", segments=7), "an even number of them"),
    (entry("kaleido", segments=1), "segments = 1: expected 2..64"),
    (entry("kaleido", mirror="yes"), "mirror is true or false"),
    (entry("kaleido", source={"centre": [0.5, 0.5]}), "expected {center, angle, radius}"),
    (entry("kaleido", source={"radius": 0}), "source radius = 0"),
    (entry("kaleido", zoom=[[0.0, 1.0], [0.5, 0.0]]), "expected positive numbers"),
    (entry("kaleido", pulse=-1.0), "pulse = -1.0"),
    (entry("tunnel", scale=1.0), "scale = 1.0: expected 0.05..0.95"),
    (entry("tunnel", depth=2.5), "a whole number"),
    (entry("tunnel", frame={"width": 4}), "expected {color, width}"),
    (entry("tunnel", frame={"color": "nope"}), "frame color"),
    (entry("melt"), "`mode` is required"),
    (entry("melt", mode="wax"), "none of breathe, ripple, swirl, drip, noise"),
    (entry("melt", mode="drip", amount=1.5), "amount = 1.5: expected 0..1"),
    (entry("melt", mode="noise", scale=0), "scale = 0"),
    (entry("kaleido", edge={"color": "#ffffff", "width": 14}), "give the mask and its objects"),
    (entry("tunnel", mask="ground", objects="Reisen*", back="rose"), "figure mask"),
    (entry("melt", mode="swirl", mask="ground"), "needs the figure's `objects`"),
    (entry("kaleido", times=[0.5], lane="kick"), "one of them"),
    (entry("tunnel", min_db=-3), "give the `lane`"),
    (entry("kaleido", blur=3), "unknown keys ['blur']"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    pattern = "".join("\\" + c if c in "[]()." else c for c in frag)
    with pytest.raises(FX.FxError, match=pattern):
        FX.normalize(spec, DAWN)


def test_defaults_and_each_melt_mode_s_own():
    k = FX.normalize(entry("kaleido"), DAWN)
    assert (k["segments"], k["mirror"], k["source"], k["zoom"], k["times"]) == (
        6, True, {"center": None, "angle": 90.0, "radius": None}, [[0.0, 1.0]], [])
    t = FX.normalize(entry("tunnel", frame={"color": ["love", "gold"]}), DAWN)
    assert t["frame"]["width"] == RM.FRAME_WIDTH and t["frame"]["color"][1] == pytest.approx(PAL.srgb(DAWN["gold"]),
                                                                                           abs=1e-5)
    for mode in RM.MODES:
        m = FX.normalize(entry("melt", mode=mode), DAWN)
        assert {k: m[k] for k in ("amount", "scale", "speed")} == pytest.approx(RM.MELT[mode])
    assert FX.normalize(entry("melt", mode="swirl", amount=300), DAWN)["amount"] == 300.0


def test_a_mask_asks_for_the_figure_layer_and_no_mask_asks_for_nothing():
    masked = plan(entry("kaleido", mask="ground", objects="Reisen*"))
    fx = masked["fx"][0]
    assert FX.uses_figure(fx) and FX.sources(fx, F0 + 5, F0) == [] and not FX.holds(fx)
    figure = {"kind": "figure", "key": FX.figure_key(fx), "objects": ["Reisen*"], "shot": "a"}
    assert {f for f, items in TR.demands(masked).items() if figure in items} == set(range(F0, F0 + 30))
    for kind, kw in (("kaleido", {}), ("tunnel", {}), ("melt", {"mode": "noise"})):
        pl = plan(entry(kind, **kw))
        assert not FX.uses_figure(pl["fx"][0]) and TR.demands(pl) == {}


def test_each_output_may_have_its_own_wedges():
    pl = plan(entry("kaleido", segments=6, aspect={"t": {"segments": 12}}))
    assert pl["fx"][0]["segments"] == 6 and FX.for_output(pl["fx"][0], "t")["segments"] == 12
    img = texture(3)
    mine = runner(pl, output="t").frame(F0 + 3, img)
    assert np.allclose(mine, runner(plan(entry("kaleido", segments=12))).frame(F0 + 3, img))
    assert not np.allclose(mine, runner(pl).frame(F0 + 3, img))


def test_a_centre_on_a_scene_point_is_projected_on_every_frame():
    pl = plan(entry("tunnel", center={"point": "bone('head').head"}, mask="ground", objects="Reisen*"))
    fx = pl["fx"][0]
    assert FX.points(fx) == [(FX.point_key(fx), "bone('head').head")]
    point = {"kind": "point", "key": FX.point_key(fx), "expr": "bone('head').head", "shot": "a"}
    assert {f for f, items in TR.demands(pl).items() if point in items} == set(range(F0, F0 + 30))
    img = texture(4)
    ffx = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None,
                     point_of=lambda key, g: ((20.0 + g, 30.0), None), figure_of=lambda key, g: np.zeros((H, W)))
    here = RM.tunnel_levels((W, H), (25.0, 30.0), 0.5, 0.0, RM.tunnel_phase(fx, 4 / FPS, []), 10)
    assert np.allclose(ffx.frame(F0 + 4, img), RMF.tunnel(img, here, (25.0, 30.0)), atol=1e-6)


# ---------------------------------------------------------------- the kaleidoscope's wedges
def turned(x, y, c, deg):
    """(x, y) px turned `deg` counter-clockwise on screen about c."""
    a = math.radians(deg)
    dx, dy = x - c[0], c[1] - y
    return c[0] + dx * math.cos(a) - dy * math.sin(a), c[1] - (dx * math.sin(a) + dy * math.cos(a))


def reflected(x, y, c, deg):
    """(x, y) px reflected in the line through c pointing `deg` counter-clockwise from the right."""
    a = math.radians(deg)
    dx, dy = x - c[0], c[1] - y
    ux, uy = math.cos(a), math.sin(a)
    d = dx * ux + dy * uy
    rx, ry = 2 * d * ux - dx, 2 * d * uy - dy
    return c[0] + rx, c[1] - ry


def coords(x, y, n=6, mirror=True, axis=90.0, turn=0.0, spin=0.0, src=(48.0, 32.0), zoom=1.0, radius=None):
    sx, sy = RM.kaleido_coords(np.float32([x]), np.float32([y]), (48.0, 32.0), n, mirror, axis, turn, spin, src, zoom,
                               radius)
    return float(sx[0]), float(sy[0])


def test_the_first_wedge_is_the_picture_as_it_is_and_the_others_its_mirrors_and_turns():
    rng = np.random.default_rng(1)
    c = (48.0, 32.0)
    for n in (6, 8, 12):
        a = 360.0 / n
        for _ in range(20):
            r, th = rng.uniform(3, 30), rng.uniform(90 - a / 2 + 1, 90 + a / 2 - 1)    # inside the first wedge
            x, y = c[0] + r * math.cos(math.radians(th)), c[1] - r * math.sin(math.radians(th))
            assert coords(x, y, n) == pytest.approx((x, y), abs=1e-3)
            for k in range(1, n):          # every second wedge turned, the ones between mirrored about the edges
                assert coords(*turned(x, y, c, 2 * a * k), n) == pytest.approx((x, y), abs=2e-3)
            assert coords(*reflected(x, y, c, 90 + a / 2), n) == pytest.approx((x, y), abs=2e-3)
            assert coords(*reflected(x, y, c, 90 - a / 2), n) == pytest.approx((x, y), abs=2e-3)
            plain = coords(*turned(x, y, c, a), n, mirror=False)      # without mirrors: each next wedge turned
            assert plain == pytest.approx((x, y), abs=2e-3)
            # across a plain copy's edge the copy is turned, not mirrored: the reflected point reads the original's
            # reflection about the wedge's middle
            assert coords(*reflected(x, y, c, 90 + a / 2), n, mirror=False) == pytest.approx(
                reflected(x, y, c, 90.0), abs=2e-3)


def test_every_pixel_reads_the_source_wedge_turned_and_zoomed():
    xs, ys = RM.grid((W, H))
    src = (20.0, 40.0)
    for n, turn in ((6, 0.0), (8, 25.0), (12, -40.0)):
        sx, sy = RM.kaleido_coords(xs, ys, (48.0, 32.0), n, True, 30.0, turn, 7.0, src, 2.0)
        ang = np.degrees(np.arctan2(src[1] - sy, sx - src[0]))
        off = (ang - (30.0 + turn) + 180.0) % 360.0 - 180.0
        r_out = np.hypot(xs - 48.0, ys - 32.0)
        r_src = np.hypot(sx - src[0], sy - src[1])
        far = r_out > 2.0
        assert np.all(np.abs(off[far]) <= 180.0 / n + 1e-2)                    # inside the wedge
        assert np.allclose(r_src, r_out / 2.0, atol=1e-3)                      # zoom 2: half as far out
    sx, sy = RM.kaleido_coords(xs, ys, (48.0, 32.0), 6, True, 90.0, 0.0, 0.0, (48.0, 32.0), 1.0, radius=10.0)
    r_src = np.hypot(sx - 48.0, sy - 32.0)
    assert r_src.max() <= 10.0 + 1e-3 and np.allclose(r_src[np.hypot(xs - 48, ys - 32) < 9.5],
                                                      np.hypot(xs - 48, ys - 32)[np.hypot(xs - 48, ys - 32) < 9.5],
                                                      atol=1e-3)               # rings: folded back past the radius


def test_the_spin_turns_the_whole_kaleidoscope_and_the_turn_only_what_it_shows():
    c = (48.0, 32.0)
    x, y = 60.0, 12.0
    assert coords(*turned(x, y, c, 25.0), spin=25.0) == pytest.approx(coords(x, y), abs=2e-3)
    turned_src = coords(x, y, turn=25.0)
    assert turned_src == pytest.approx(turned(*coords(x, y), c, 25.0), abs=2e-3)


def test_the_source_steps_on_each_hit_eased_and_holds_between():
    fx = FX.normalize(entry("kaleido", to=2.0, turn=0, step=40, glide=0.3, times=[0.5, 1.2]), DAWN)
    times = fx["times"]
    assert RM.kaleido_turn(fx, 0.49, times) == 0.0
    assert RM.kaleido_turn(fx, 0.9, times) == pytest.approx(40.0)
    assert RM.kaleido_turn(fx, 1.9, times) == pytest.approx(80.0)
    ts = np.arange(0.0, 2.0, 1.0 / FPS)
    turns = np.array([RM.kaleido_turn(fx, t, times) for t in ts])
    assert np.all(np.diff(turns) >= 0) and np.diff(turns).max() < 40.0 * 0.25   # no jump: it glides over frames
    spun = FX.normalize(entry("kaleido", to=2.0, turn=12, spin=0), DAWN)
    assert RM.kaleido_turn(spun, 1.5, []) == pytest.approx(18.0)


def test_a_hit_turns_the_picture_the_kaleidoscope_shows():
    pl = plan(entry("kaleido", to=2.0, turn=0, step=60, glide=0.2, times=[0.5]))
    img = texture(5)
    ffx = runner(pl)
    before, after = ffx.frame(F0 + 10, img), ffx.frame(F0 + 40, img)
    want = RMF.kaleido(img, RMF.half_map((W, H), lambda xs, ys: RM.kaleido_coords(
        xs, ys, (47.5, 31.5), 6, True, 90.0, 60.0, 0.0, (47.5, 31.5), 1.0)), 1.0)
    assert np.allclose(after, want, atol=1e-5) and not np.allclose(before, after, atol=0.05)


def test_a_zoom_pulses_on_a_hit_and_follows_its_keys():
    fx = FX.normalize(entry("kaleido", to=2.0, zoom=[[0.0, 1.0], [1.0, 2.0]], pulse=0.5, decay=0.2, times=[0.5]),
                      DAWN)
    assert RM.kaleido_zoom(fx, 0.25, fx["times"]) == pytest.approx(1.25)
    assert RM.kaleido_zoom(fx, 0.5 + RM.ATTACK, fx["times"]) == pytest.approx((1.5 + RM.ATTACK) * 1.5, rel=1e-3)
    assert RM.kaleido_zoom(fx, 1.9, fx["times"]) == pytest.approx(2.0, rel=1e-3)


def test_plain_copies_are_blended_across_their_seams():
    pl = plan(entry("kaleido", mirror=False, segments=5, turn=0))
    img = ramp()
    out = runner(pl).frame(F0, img)
    xs, ys = RM.grid((W, H))
    hard = RMF.kaleido(img, RM.kaleido_coords(xs, ys, (47.5, 31.5), 5, False, 90.0, 0.0, 0.0, (47.5, 31.5), 1.0), 1.0)
    d, _ = RM.kaleido_seams(xs, ys, (47.5, 31.5), 5, 90.0, 0.0)
    on_seam = (d < 0.4) & (np.hypot(xs - 47.5, ys - 31.5) > 6)
    away = d > 1.5
    assert np.allclose(out[away], hard[away], atol=1e-5)
    assert np.abs(out[on_seam] - hard[on_seam]).max() > 0.02          # the jump is softened where the copies meet


# ---------------------------------------------------------------- the tunnel
def test_a_level_down_is_the_same_picture_and_a_level_fallen_through_is_where_it_began():
    size, c = (W, H), (40.0, 30.0)
    rng = np.random.default_rng(2)
    px, py = rng.uniform(0, W - 1, 400), rng.uniform(0, H - 1, 400)
    for scale, twist in ((0.5, 0.0), (0.6, 20.0), (0.7, -35.0)):
        for phase in (0.0, 0.3, 0.85):
            sx, sy, lev = RM.tunnel_source(px, py, size, c, scale, twist, phase, 12)
            # the point one level in (shrunk `scale` times and turned `twist` about the centre) reads the same
            a = math.radians(twist)
            dx, dy = px - c[0], py - c[1]
            ix = c[0] + scale * (dx * math.cos(a) + dy * math.sin(a))
            iy = c[1] + scale * (-dx * math.sin(a) + dy * math.cos(a))
            tx, ty, tl = RM.tunnel_source(ix, iy, size, c, scale, twist, phase, 12)
            ok = (lev >= 0) & (lev < 10)
            assert ok.sum() > 300
            assert np.allclose(tx[ok], sx[ok], atol=1e-6) and np.allclose(ty[ok], sy[ok], atol=1e-6)
            assert np.all(tl[ok] == lev[ok] + 1)
        # falling one whole level brings it back: just before is just after
        b = RM.tunnel_source(px, py, size, c, scale, twist, 1.0 - 1e-7, 12)
        e = RM.tunnel_source(px, py, size, c, scale, twist, 1.0, 12)
        same = (b[2] - 1 == e[2]) & (e[2] < 10)
        assert same.mean() > 0.97 and np.allclose(b[0][same], e[0][same], atol=1e-3)


def test_level_zero_is_the_frame_itself_and_the_deepest_fades_in():
    size, c = (W, H), (47.5, 31.5)
    sx, sy, lev = RM.tunnel_source(np.float64([2.0, 90.0]), np.float64([3.0, 60.0]), size, c, 0.5, 0.0, 0.0, 8)
    assert lev.tolist() == [0, 0] and np.allclose(sx, [2.0, 90.0]) and np.allclose(sy, [3.0, 60.0])
    levels = RM.tunnel_levels((1920, 1080), (959.5, 539.5), 0.5, 0.0, 3.25, 6)
    assert levels[0][:2] == (0, pytest.approx(0.5 ** -0.25)) and levels[-1][0] == 7
    assert levels[-1][3] == pytest.approx(0.25) and all(lv[3] == 1.0 for lv in levels[:-1])
    deep = RM.tunnel_levels((1920, 1080), (959.5, 539.5), 0.5, 0.0, 0.0, 30)
    assert deep[-1][1] * 1080 >= 1.0 > deep[-1][1] * 0.5 * 1080          # nothing under a pixel is drawn
    twisted = RM.tunnel_levels((1920, 1080), (959.5, 539.5), 0.9, 40.0, 0.5, 4)
    assert twisted[0][0] < 0                                               # a level round it fills the corners


def test_frame_lines_keep_their_colour_as_they_fall_and_a_hit_pushes_a_level():
    cols = ["a", "b", "c"]
    for k in range(-1, 6):
        assert RM.frame_colour(cols, k, 2.999) == RM.frame_colour(cols, k - 1, 3.0)
    fx = FX.normalize(entry("tunnel", to=2.0, speed=0.5, step=1, glide=0.2, times=[1.0]), DAWN)
    assert RM.tunnel_phase(fx, 0.9, fx["times"]) == pytest.approx(0.45)
    assert RM.tunnel_phase(fx, 1.5, fx["times"]) == pytest.approx(1.75)
    phases = [RM.tunnel_phase(fx, t, fx["times"]) for t in np.arange(0, 2, 1 / FPS)]
    assert np.diff(phases).max() < 0.5                                     # pushed over frames, not cut


def test_the_drawn_tunnel_reads_where_the_maths_says():
    pl = plan(entry("tunnel", to=2.0, scale=0.6, twist=15, speed=0.4, depth=8))
    img = ramp()
    f = F0 + 20
    out = runner(pl).frame(f, img)
    phase = RM.tunnel_phase(pl["fx"][0], (f - F0) / FPS, [])
    xs, ys = RM.grid((W, H))

    def source(dx, dy):
        return RM.tunnel_source(xs.astype(float) + dx, ys.astype(float) + dy, (W, H), (47.5, 31.5), 0.6, 15, phase, 8)
    sx, sy, lev = source(0.0, 0.0)
    # away from any level's edge (its own and the next one's, a pixel and a half), on the levels large enough to be
    # read without their blur
    away = (lev >= 0) & (lev <= 2)
    for dx in (-1.5, 0.0, 1.5):
        for dy in (-1.5, 0.0, 1.5):
            away &= source(dx, dy)[2] == lev
    away &= (sx > 1.0) & (sx < W - 2.0) & (sy > 1.0) & (sy < H - 2.0)
    want = np.dstack([sx / (W - 1), sy / (H - 1)])
    assert away.sum() > 2000
    assert np.abs(out[..., :2][away] - want[away]).max() < 0.02


# ---------------------------------------------------------------- melting
SIZE, C, UNIT = (160, 120), (79.5, 59.5), 120.0


def field(mode, amount, age=1.3, scale=None, speed=None, seed=(1, 0), front=None, level=None):
    d = RM.MELT[mode]
    return RM.melt_field(mode, SIZE, C, level or (lambda a: amount), d["scale"] if scale is None else scale,
                         d["speed"] if speed is None else speed, age, UNIT, seed, front)


@pytest.mark.parametrize("mode", RM.MODES)
def test_nothing_moves_at_amount_zero(mode):
    dx, dy = field(mode, 0.0)
    assert not dx.any() and not dy.any() and dx.shape == (SIZE[1], SIZE[0])


@pytest.mark.parametrize("mode, amount", [("breathe", 0.3), ("breathe", -0.4), ("ripple", 0.05), ("swirl", 300.0),
                                          ("drip", 0.4), ("noise", 0.05)])
def test_the_displacement_is_bounded_by_the_amount(mode, amount):
    xs, ys = RM.grid(SIZE)
    r = np.hypot(xs - C[0], ys - C[1])
    for age in (0.0, 0.4, 1.3, 2.9):
        dx, dy = field(mode, amount, age)
        d = np.hypot(dx, dy)
        if mode == "breathe":       # a share of the distance from the centre, at most a / (1 + a) of it
            g = abs(amount) / (1.0 - abs(amount)) if amount < 0 else amount / (1.0 + amount)
            assert np.all(d <= g * r + 1e-3)
        elif mode == "swirl":       # a turn about the centre: the chord of at most `amount` degrees
            assert np.all(d <= 2 * r * math.sin(min(math.radians(amount), math.pi) / 2) + 1e-3)
            assert np.allclose(np.hypot(xs + dx - C[0], ys + dy - C[1]), r, atol=1e-3)      # it keeps the radius
        elif mode == "drip":        # straight down from above, never further than the reach
            assert not dx.any() and np.all(dy <= 0) and np.all(-dy <= amount * UNIT + 1e-3)
            src = ys + dy
            assert np.all(np.diff(src, axis=0) >= -1e-3)                     # it stretches, it does not fold
        else:
            assert d.max() <= amount * UNIT + 1e-3
        assert d.max() > 0 or (mode == "breathe" and age == 0.0)


def test_melts_move_smoothly_from_frame_to_frame_and_by_their_seed():
    for mode, amount in (("noise", 0.05), ("drip", 0.3), ("ripple", 0.05), ("breathe", 0.3), ("swirl", 200.0)):
        speed = 0.5 if mode == "swirl" else None                             # a held swirl (speed 0) stands still
        a, b = field(mode, amount, 1.3, speed=speed), field(mode, amount, 1.3 + 1.0 / FPS, speed=speed)
        step = max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max())
        assert 0 < step < 0.25 * max(np.abs(a[0]).max(), np.abs(a[1]).max()) + 1e-6, mode
    one, two = field("noise", 0.05, seed=(1, 0)), field("noise", 0.05, seed=(1, 0))
    assert np.array_equal(one[0], two[0]) and not np.allclose(one[0], field("noise", 0.05, seed=(2, 0))[0])
    assert not np.allclose(field("drip", 0.3, seed=(1, 0))[1], field("drip", 0.3, seed=(1, 1))[1])


def test_a_ripple_on_a_hit_runs_out_from_the_centre():
    fx = FX.normalize(entry("melt", to=3.0, mode="ripple", amount=0.05, decay=0.5, times=[1.0]), DAWN)

    def level(age):
        return RM.melt_level(fx, age, fx["times"], lambda t: 1.0)
    front = RM.FRONT * UNIT
    assert not field("ripple", 0, age=0.9, level=level, front=front)[0].any()        # before the hit, still
    xs, ys = RM.grid(SIZE)
    r = np.hypot(xs - C[0], ys - C[1])
    dx, dy = field("ripple", 0, age=1.3, level=level, front=front)
    moved = np.hypot(dx, dy) > 1e-4
    assert moved.any() and r[moved].max() <= front * 0.3 + 2.0                     # not past its front yet
    assert np.hypot(*field("ripple", 0.05, age=1.3)).max() > 0                     # steady: everywhere at once


def test_the_amount_swells_on_hits_and_dies_away_or_follows_the_strength():
    fx = FX.normalize(entry("melt", to=3.0, mode="swirl", amount=100, decay=0.3, times=[1.0]), DAWN)
    assert RM.melt_level(fx, 0.9, fx["times"], lambda t: 1.0) == 0.0
    assert RM.melt_level(fx, 1.0 + RM.ATTACK, fx["times"], lambda t: 1.0) == pytest.approx(100.0)
    assert RM.melt_level(fx, 1.3 + RM.ATTACK, fx["times"], lambda t: 1.0) == pytest.approx(100 * math.exp(-1), rel=1e-6)
    steady = FX.normalize(entry("melt", mode="swirl", amount=100), DAWN)
    assert RM.melt_level(steady, 0.4, [], lambda t: 0.5) == 50.0


def test_melt_strength_scales_the_amount_rather_than_mixing():
    img = texture(7)
    half = runner(plan(entry("melt", mode="swirl", amount=80, strength=0.5))).frame(F0 + 5, img)
    want = runner(plan(entry("melt", mode="swirl", amount=40))).frame(F0 + 5, img)
    mixed = img * 0.5 + runner(plan(entry("melt", mode="swirl", amount=80))).frame(F0 + 5, img) * 0.5
    assert np.allclose(half, want, atol=1e-5) and not np.allclose(half, mixed, atol=1e-3)
    still = runner(plan(entry("melt", mode="swirl", amount=80, strength=0.0)))
    assert still.frame(F0 + 5, img) is img


# ---------------------------------------------------------------- masks
def test_a_ground_mask_keeps_the_figure_exactly_as_it_was():
    img = texture(11)
    m = disc(48, 40, 14)
    for spec in (entry("kaleido", turn=30), entry("tunnel", speed=0.8), entry("melt", mode="swirl", amount=200)):
        pl = plan({**spec, "mask": "ground", "objects": "Reisen*"})
        out = runner(pl, figure=m).frame(F0 + 12, img)
        inside, outside = m >= 1.0, disc(48, 40, 18) == 0
        assert np.array_equal(out[inside], img[inside]), spec["kind"]
        assert np.abs(out[outside] - img[outside]).mean() > 0.02, spec["kind"]


def test_the_figure_s_place_is_filled_from_the_ground_round_it():
    ground = np.zeros((H, W, 3), np.float32)
    ground[..., 2] = np.linspace(0.2, 0.8, W, dtype=np.float32)[None, :]              # blue, rising to the right
    m = disc(48, 32, 10)
    img = ground * (1 - m[..., None]) + np.float32([1, 0, 0]) * m[..., None]         # a red figure
    hole = (m > 0).astype(np.float32)
    out = RMF.fill(img, hole)
    assert np.array_equal(out[hole == 0], img[hole == 0])                            # the ground as it was
    assert out[..., 0][hole > 0].max() < 0.02                                        # no red left
    assert np.allclose(out[..., 2][hole > 0], ground[..., 2][hole > 0], atol=0.12)   # the ground mirrored in
    edge = np.zeros((H, W), np.float32)
    edge[:, :20] = 1.0                                       # a run at the frame's edge: mirrored from its one side
    assert np.allclose(RMF.fill(ground, edge)[:, :20], ground[:, 39:19:-1])
    rows = np.zeros((H, W), np.float32)
    rows[20:30] = 1.0                                        # whole rows: no ground beside them, spread in instead
    spread = RMF.fill(ground, rows)
    assert np.isfinite(spread).all() and np.allclose(spread[20:30, 16:-16, 2], ground[20:30, 16:-16, 2], atol=0.05)
    assert np.allclose(spread[20:30, :, 2], ground[20:30, :, 2], atol=0.1)          # softer toward the frame's sides


def test_a_figure_mask_remaps_her_alone_over_her_filled_place_or_a_flat_back():
    ground = np.full((H, W, 3), [0.1, 0.5, 0.2], np.float32)
    m = disc(48, 20, 9)
    img = ground * (1 - m[..., None]) + np.float32([0.9, 0.2, 0.6]) * m[..., None]
    pl = plan(entry("melt", mode="drip", amount=0.4, mask="figure", objects="Reisen*", back="rose"))
    out = runner(pl, figure=m).frame(F0 + 10, img)
    rose = np.float32(PAL.srgb(DAWN["rose"]))
    far = disc(48, 20, 9 + 0.4 * 64) == 0
    assert np.allclose(out[far], rose, atol=1e-5)                    # nothing but the back where she cannot reach
    assert np.isclose(out[..., 0], 0.9, atol=0.02).sum() > 50        # her colour, moved
    kept = runner(plan(entry("melt", mode="drip", amount=0.4, mask="figure", objects="Reisen*")), figure=m)
    out = kept.frame(F0 + 10, img)
    assert np.allclose(out[:4], ground[:4], atol=1e-5)               # above her the ground stays as it was


def test_a_figure_tunnel_is_a_queue_of_whole_copies_far_to_near():
    q = RM.tunnel_queue((1920, 1080), (959.5, 539.5), 0.5, 10.0, 2.25, 6)
    assert [lv[0] for lv in q] == [7, 6, 5, 4, 3, 2, 1, 0]                           # far first
    assert q[-1][3] == pytest.approx(0.75) and q[0][3] == pytest.approx(0.25)        # the nearest fades as it passes
    pl = plan(entry("tunnel", speed=0.5, mask="figure", objects="Reisen*", back="#000000"))
    m = (disc(48, 32, 6) > 0.5).astype(np.float32)          # a hard edge: a soft one is drawn as a collage draws it
    img = np.dstack([m, m, m]).astype(np.float32)
    out = runner(pl, figure=m).frame(F0, img)                                        # phase 0: her as rendered
    assert np.allclose(out, img, atol=2e-3)
    later = runner(pl, figure=m).frame(F0 + 20, img)                                 # her copies, smaller, behind
    assert later[31, 47].min() > 0.9 and later[..., 0].sum() > 0.5 * img[..., 0].sum()


def test_strength_mixes_a_kaleidoscope_and_a_tunnel_and_a_rect_keeps_them_in():
    img = texture(13)
    for kind in ("kaleido", "tunnel"):
        full = runner(plan(entry(kind))).frame(F0 + 6, img)
        half = runner(plan(entry(kind, strength=0.5))).frame(F0 + 6, img)
        assert np.allclose(half, img * 0.5 + full * 0.5, atol=1e-5)
        boxed = runner(plan(entry(kind, rect=[0.0, 0.0, 0.5, 1.0]))).frame(F0 + 6, img)
        assert np.array_equal(boxed[:, W // 2 + 1:], img[:, W // 2 + 1:])
