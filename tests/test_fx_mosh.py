"""The datamosh ([[fx]] kind "mosh", mkmmd.core.framefx and mkmmd.framefx) with a made-up `flow` and a figure `mask`: its
keys, the frames and layers it reads, the block moves of each flow (seeded, whole pixels, stuck blocks in patches), a still
moshed from its own picture, the mask that leaves the figure exactly as it was and keeps her colours out of the ground, the
rect, and the cut's own motion as it always was without a flow."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
MOON = PAL.get("rose-pine-moon")
W, H = 96, 64
PX = min(W, H) / 1080.0                 # the test frames' pixels per pixel at 1080
BLOCK = 8 / PX                          # 8 px blocks in them
STILL = 1 / FPS                         # a one-frame clip


def px(n):
    """`n` px of the test frames, in px at 1080 (what `amount` is given in)."""
    return n / PX


def plan(*fx, duration=2.0, size=(W, H)):
    data = {"project": {"duration": duration}, "output": [{"name": "t", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": duration}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, MOON)


def mosh(**kw):
    return {"kind": "mosh", "from": 0.0, "to": 1.0, **kw}


def still(**kw):
    """A one-frame clip moshed with `kw`."""
    return plan(mosh(to=STILL, block=BLOCK, **kw), duration=STILL)


def texture(seed=1, size=(W, H)):
    """Smooth colour noise: what a block carries can be told from what it left."""
    import cv2
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((size[1], size[0], 3)).astype(np.float32), (0, 0), 1.5)
    return (t - t.min()) / (t.max() - t.min())


def runner(pl, frames=None, figure=None, size=(W, H)):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: (frames or {}).get(g), point_of=None,
                      figure_of=None if figure is None else (lambda key, g: figure))


def square(x0, x1, y0, y1, size=(W, H)):
    m = np.zeros((size[1], size[0]), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


# ---------------------------------------------------------------- the entries
@pytest.mark.parametrize("spec, frag", [
    ({"flow": "drift"}, "expected a table whose `kind` is one of drift, zoom, swirl, noise"),
    ({"flow": {"kind": "melt"}}, "one of drift, zoom, swirl, noise"),
    ({"flow": {"kind": "zoom", "angle": 30}}, "flow (zoom): unknown keys ['angle']"),
    ({"flow": {"kind": "drift", "center": [0.5, 0.5]}}, "flow (drift): unknown keys ['center']"),
    ({"flow": {"kind": "noise", "steps": 0}}, "flow steps = 0"),
    ({"flow": {"kind": "noise", "steps": 2.5}}, "a whole number"),
    ({"flow": {"kind": "noise", "stuck": 1.5}}, "flow stuck = 1.5"),
    ({"flow": {"kind": "noise", "jitter": -1}}, "flow jitter = -1"),
    ({"flow": {"kind": "noise", "size": 0.01}}, "flow size = 0.01"),
    ({"flow": {"kind": "swirl", "center": 0.5}}, "expected [x, y]"),
    ({"mask": "ground"}, "needs the figure's `objects`"),
    ({"objects": "Carly*"}, "give both or neither"),
    ({"mask": "sky", "objects": "Carly*"}, "none of figure, ground"),
    ({"mask": "ground", "objects": [""]}, "expected object patterns"),
])
def test_bad_moshes_say_what_is_wrong(spec, frag):
    pattern = "".join("\\" + c if c in "[]()" else c for c in frag)
    with pytest.raises(FX.FxError, match=pattern):
        FX.normalize(mosh(**spec), MOON)


def test_a_flow_takes_its_defaults_and_its_kind_s_own_key():
    fx = FX.normalize(mosh(flow={"kind": "drift", "amount": 4, "angle": 30}), MOON)
    assert fx["flow"] == {"kind": "drift", "amount": 4.0, "steps": 12, "jitter": 0.25, "stuck": 0.3, "seed": 1,
                          "angle": 30.0}
    assert FX.normalize(mosh(flow={"kind": "zoom"}), MOON)["flow"]["center"] == [0.5, 0.5]
    assert FX.normalize(mosh(flow={"kind": "noise"}), MOON)["flow"]["size"] == 0.3
    plain = FX.normalize(mosh(), MOON)
    assert plain["flow"] is None and plain["mask"] is None and "objects" not in plain


def test_a_flow_reads_its_first_frame_and_a_mask_its_figure_there_and_on_each_frame():
    pl = plan(mosh(flow={"kind": "noise"}, mask="ground", objects="Carly*"))
    fx = pl["fx"][0]
    assert FX.sources(fx, F0 + 10, F0) == [F0] and FX.uses_figure(fx) and FX.holds(fx)
    figure = {"kind": "figure", "key": FX.figure_key(fx), "objects": ["Carly*"], "shot": "a"}
    assert {f for f, items in TR.demands(pl).items() if figure in items} == set(range(F0, F0 + 30))
    assert TR.fx_frames(pl, [F0 + 10]) == {F0: [{"kind": "plate", "shot": "a"}, figure]}   # what mk look renders beside
    assert TR.fx_frames(pl, [F0]) == {}                        # a still reads nothing beside itself
    cut = plan(mosh(block=8))["fx"][0]                         # without a flow: the cut's frames, as before
    assert FX.sources(cut, F0 + 3, F0) == [F0, F0 + 1, F0 + 2, F0 + 3] and not FX.holds(cut)
    assert not FX.uses_figure(cut) and TR.demands(plan(mosh())) == {}


# ---------------------------------------------------------------- the block moves
def flow(**kw):
    return FX.normalize(mosh(flow=kw), MOON)["flow"]


def test_the_same_seed_moves_the_same_blocks_and_another_seed_others():
    f = flow(kind="noise", amount=8, seed=4)
    a, b = FF.mosh_flow(f, (W, H), 8.0, 1.0, 0), FF.mosh_flow(f, (W, H), 8.0, 1.0, 0)
    assert a[1] == b[1] == 8 and a[0].shape == (8, 12, 2) and a[0].dtype == np.int32
    assert np.array_equal(a[0], b[0])
    assert not np.array_equal(a[0], FF.mosh_flow(dict(f, seed=5), (W, H), 8.0, 1.0, 0)[0])
    assert not np.array_equal(a[0], FF.mosh_flow(f, (W, H), 8.0, 1.0, 1)[0])     # another entry, another take
    img = texture(2)
    assert np.array_equal(FF.mosh_carry(img, a[0], 8, 5), FF.mosh_carry(img, b[0], 8, 5))


def test_each_kind_moves_its_blocks_its_own_way():
    calm = {"jitter": 0.0, "stuck": 0.0}
    drift, b = FF.mosh_flow(flow(kind="drift", amount=5, angle=90, **calm), (W, H), 8.0, 2.0, 0)
    assert np.all(drift == [0, -10])                          # up, 5 px at 1080 of 2 px each
    middles = (np.arange(20) + 0.5) * 8 - 0.5 - 79.5           # the blocks' middles about the frame's middle
    dx, dy = np.meshgrid(middles, middles)
    far = np.hypot(dx, dy) > 40
    zoom, _ = FF.mosh_flow(flow(kind="zoom", amount=20, **calm), (160, 160), 8.0, 1.0, 0)
    assert np.all((zoom[..., 0] * dx + zoom[..., 1] * dy)[far] > 0)               # out from the middle
    corner = 20 * np.hypot(76, 76) / np.hypot(80, 80)          # the farthest move is at the frame's corner
    assert np.hypot(zoom[..., 0], zoom[..., 1]).max() == pytest.approx(corner, abs=0.75)
    swirl, _ = FF.mosh_flow(flow(kind="swirl", amount=20, **calm), (160, 160), 8.0, 1.0, 0)
    along = (swirl[..., 0] * dy - swirl[..., 1] * dx) / (np.hypot(swirl[..., 0], swirl[..., 1]) * np.hypot(dx, dy))
    assert np.all(along[far] > 0.95)                           # round the middle, counter-clockwise on screen
    assert swirl[0, 10, 0] < 0 and swirl[10, 19, 1] < 0      # the top moves left, the right side up
    noise, _ = FF.mosh_flow(flow(kind="noise", amount=12, **calm), (160, 160), 8.0, 1.0, 0)
    assert np.hypot(noise[..., 0], noise[..., 1]).max() == pytest.approx(12, abs=0.75)
    assert len({tuple(v) for v in noise.reshape(-1, 2)}) > 20                  # a field, not one move


def test_a_share_of_the_blocks_is_stuck_in_patches_and_more_of_it_holds_the_same_ones():
    f = flow(kind="drift", amount=6, jitter=0.0, stuck=0.25, seed=2)
    some, _ = FF.mosh_flow(f, (160, 160), 8.0, 1.0, 0)
    more, _ = FF.mosh_flow(dict(f, stuck=0.5), (160, 160), 8.0, 1.0, 0)
    a, b = ~some.any(-1), ~more.any(-1)
    assert a.sum() == 100 and b.sum() == 200                   # of 20 x 20 blocks
    assert np.all(b[a])                                        # nested: the same patches and more
    lone = sum(1 for j in range(20) for i in range(20) if a[j, i] and not any(
        a[jj, ii] for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)) if 0 <= jj < 20 and 0 <= ii < 20))
    assert lone < 10                                           # patches, not scattered blocks
    jit, _ = FF.mosh_flow(dict(f, jitter=0.5), (160, 160), 8.0, 1.0, 0)
    assert np.array_equal(~jit.any(-1), a)                     # jitter leaves the stuck blocks where they were


def test_a_frame_of_another_size_is_cut_into_the_same_blocks_and_moshed_the_same_way():
    f = flow(kind="drift", amount=6, angle=20, jitter=0.2, stuck=0.3, seed=7)
    small, _ = FF.mosh_flow(f, (240, 160), 9.5, 1.0, 0)       # mk look's size, say, and mk post's twice as large
    big, _ = FF.mosh_flow(f, (480, 320), 19.0, 2.0, 0)
    assert small.shape == big.shape == (17, 26, 2)
    assert np.array_equal(small.any(-1), big.any(-1))         # the same blocks stuck
    assert np.abs(big - 2 * small).max() <= 1                  # and the same moves, rounded at each size


def test_blocks_move_whole_pixels_in_chunks_and_a_moving_block_smears_the_edge_of_a_stuck_one():
    img = texture(3)
    moves, b = FF.mosh_flow(flow(kind="noise", amount=6, jitter=0.4, stuck=0.3, seed=9), (W, H), 8.0, 1.0, 0)
    assert moves.any()
    one = FF.mosh_carry(img, moves, b, 1)
    for j in range(H // 8):
        for i in range(W // 8):
            vx, vy = moves[j, i]
            ys, xs = np.mgrid[j * 8:(j + 1) * 8, i * 8:(i + 1) * 8]
            src = img[np.clip(ys - vy, 0, H - 1), np.clip(xs - vx, 0, W - 1)]
            assert np.array_equal(one[j * 8:(j + 1) * 8, i * 8:(i + 1) * 8], src)    # a sharp copy, block by block
    row = np.zeros((1, 2, 2), np.int32)
    row[0, 1] = (3, 0)                                         # a block that moves right beside one that stays
    strip = np.repeat(np.linspace(0, 1, 16, dtype=np.float32)[None, :, None], 3, 2)
    out = FF.mosh_carry(strip, row, 8, 4)
    assert np.array_equal(out[0, :8], strip[0, :8])            # the stuck block as it was
    assert np.array_equal(out[0, 8:11], strip[0, 5:8]) and np.array_equal(out[0, 11:14], strip[0, 5:8])  # its edge,
    assert np.array_equal(out[0, 14:16], strip[0, 5:7])                                               # again and again


def test_a_masked_flow_keeps_the_blocks_still_whose_source_reaches_the_rest():
    moves = np.zeros((1, 3, 2), np.int32)
    moves[0, :, 0] = 4                                         # every block moves 4 px to the right
    outside = np.zeros((8, 24), bool)
    outside[:, 2] = True                                       # one column of the rest, inside the first block
    assert FF.mosh_stay(moves, 8, outside)[0, :, 0].tolist() == [0, 4, 4]   # the block whose source holds it stays
    outside[:, 9] = True                                       # and one the second block reads 4 px behind it
    assert FF.mosh_stay(moves, 8, outside)[0, :, 0].tolist() == [0, 0, 4]


# ---------------------------------------------------------------- the pictures
def test_a_still_is_moshed_from_its_own_picture_and_the_same_seed_draws_it_the_same():
    img = texture(4)
    pl = still(flow={"kind": "noise", "amount": px(4), "steps": 6, "seed": 3})
    out = runner(pl, {F0: img}).frame(F0, img)
    assert not np.allclose(out, img, atol=1e-3)
    assert np.array_equal(out, runner(pl, {F0: img}).frame(F0, img))
    other = still(flow={"kind": "noise", "amount": px(4), "steps": 6, "seed": 4})
    assert not np.array_equal(out, runner(other, {F0: img}).frame(F0, img))
    assert np.array_equal(runner(still(), {F0: img}).frame(F0, img), img)   # the cut's own motion: a still has none


def test_the_held_picture_is_the_frame_as_it_reached_the_mosh_and_steps_on_each_frame():
    tinted = plan({"kind": "tint", "from": 0.0, "to": 1.0, "color": "love"},
                  mosh(block=BLOCK, flow={"kind": "drift", "amount": px(3), "stuck": 1.0}))   # every block stuck
    img = texture(5)
    first = runner(tinted, {F0 + g: texture(5 + g) for g in range(30)}).frame(F0, img)
    tint = tinted["fx"][0]
    assert np.array_equal(first, FF.gradient_map(img, tint["dark"], tint["color"], tint["light"]))
    pl = plan(mosh(block=BLOCK, flow={"kind": "swirl", "amount": px(3), "steps": 3, "seed": 2}))
    frames = {F0 + g: texture(10 + g) for g in range(30)}
    run = runner(pl, frames)
    seq = [run.frame(F0 + g, frames[F0 + g]) for g in range(4)]
    late = runner(pl, frames).frame(F0 + 3, frames[F0 + 3])                    # asked out of order: worked out again
    assert np.array_equal(seq[3], late)
    moves, b = FF.mosh_flow(pl["fx"][0]["flow"], (W, H), BLOCK * PX, PX, 0)
    assert moves.any() and np.array_equal(late, FF.mosh_carry(frames[F0], moves, b, 4 * 3))  # 3 steps a frame


@pytest.mark.parametrize("mask", ["ground", "figure"])
def test_the_mask_leaves_the_figure_or_the_ground_exactly_as_it_was(mask):
    fig = square(30, 62, 10, 54)
    fig[10:54, 29] = 0.5                                       # an antialiased edge
    pl = still(flow={"kind": "noise", "amount": px(5), "steps": 8, "seed": 1}, mask=mask, objects="Carly*")
    img = texture(6)
    out = runner(pl, {F0: img}, fig).frame(F0, img)
    keep = fig == (1.0 if mask == "ground" else 0.0)
    assert np.array_equal(out[keep], img[keep])
    moshed = (fig != 0.5) & ~keep
    assert not np.allclose(out[moshed], img[moshed], atol=1e-3)
    moves, b = FF.mosh_flow(pl["fx"][0]["flow"], (W, H), BLOCK * PX, PX, 0)
    region = fig if mask == "figure" else 1.0 - fig
    held = FF.mosh_carry(img, FF.mosh_stay(moves, b, region < FF.MOSH_SOLID), b, 8)
    assert np.array_equal(out[moshed], held[moshed])           # the rest is the held picture
    assert np.allclose(out[10:54, 29], 0.5 * img[10:54, 29] + 0.5 * held[10:54, 29], atol=1e-6)   # the outline mixed


def test_a_ground_mask_keeps_her_colours_out_of_the_ground_and_the_rect_keeps_the_mosh_inside_it():
    img = np.zeros((H, W, 3), np.float32)
    img[..., 1:] = texture(7)[..., 1:]                         # a ground with no red in it
    fig = square(40, 56, 16, 48)
    img[fig > 0] = (1.0, 0.0, 0.0)                             # she is red
    f = {"kind": "drift", "amount": px(3), "angle": 0, "steps": 6, "jitter": 0.3, "stuck": 0.2}
    masked = runner(still(flow=f, mask="ground", objects="Carly*"), {F0: img}, fig).frame(F0, img)
    assert masked[fig == 0][:, 0].max() == 0.0                # nothing of her in the ground
    assert np.array_equal(masked[fig > 0], img[fig > 0]) and not np.allclose(masked, img, atol=1e-3)
    bare = runner(still(flow=f), {F0: img}).frame(F0, img)
    assert bare[fig == 0][:, 0].max() == 1.0                   # unmasked, her red is dragged out with the rest
    band = runner(still(flow=f, mask="ground", objects="Carly*", rect=[0.0, 0.0, 0.25, 1.0], feather=4),
                  {F0: img}, fig).frame(F0, img)
    assert np.array_equal(band[:, 30:], img[:, 30:]) and not np.allclose(band[:, :20], img[:, :20], atol=1e-3)
    across = runner(still(flow=f, mask="ground", objects="Carly*", rect=[0.0, 0.0, 0.5, 1.0], feather=px(6)),
                    {F0: img}, fig).frame(F0, img)
    assert np.array_equal(across[fig > 0], img[fig > 0])     # a feathered edge across her leaves her as she was


def test_strength_mixes_the_held_picture_and_zero_leaves_the_frame():
    img = texture(8)
    f = {"kind": "zoom", "amount": px(6), "steps": 4}
    full = runner(still(flow=f), {F0: img}).frame(F0, img)
    half = runner(still(flow=f, strength=0.5), {F0: img}).frame(F0, img)
    assert not np.allclose(full, img, atol=1e-3) and np.allclose(half, 0.5 * img + 0.5 * full, atol=1e-6)
    assert runner(still(flow=f, strength=0.0), {F0: img}).frame(F0, img) is img


def test_without_a_flow_the_cut_s_own_motion_carries_the_frame_before_as_it_always_did():
    """The mosh of a video span is unchanged: the frame before the span carried by mosh_step from frame to frame."""
    import cv2
    w, h = 160, 96
    tex = cv2.GaussianBlur(np.random.default_rng(1).random((h, w)).astype(np.float32), (0, 0), 3)
    tex = np.repeat(((tex - tex.min()) / (tex.max() - tex.min()))[..., None], 3, 2)
    before = np.zeros((h, w, 3), np.float32)
    before[30:60, 60:100] = 1.0
    frames = {F0 + 29: before, **{F0 + 30 + k: np.roll(tex, 3 * k, 1) for k in range(4)}}
    data = {"project": {"duration": 2.0}, "output": [{"name": "t", "size": [w, h]}],
            "shot": [{"name": "a", "from": 0.0, "to": 1.0}, {"name": "b", "from": 1.0, "to": 2.0}],
            "fx": [{"kind": "mosh", "from": 1.0, "to": 1.2, "block": 90, "strength": 0.75}]}
    pl = TR.plan(data, FPS, F0, MOON)
    run = FF.FrameFx(pl["fx"], (w, h), fps=FPS, frame0=F0, frame_of=frames.get, point_of=None)
    held, block = before, 90 * min(w, h) / 1080.0
    for k in range(4):
        f = F0 + 30 + k
        if k:
            held = FF.mosh_step(held, frames[f - 1], frames[f], block)
        assert np.array_equal(run.frame(f, frames[f]), (frames[f] * 0.25 + held * 0.75).astype(np.float32))
    assert not np.allclose(held, before)                       # the held picture did move
