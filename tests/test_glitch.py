"""A glitch (mkmmd.core.glitch): the figure found against the bare frame, broken up in jumping slices and colour ghosts over
its window and gone on its last frame; repeatable, and never touching the picture away from the figure."""
import numpy as np

from mkmmd.core import glitch as GL

H, W, N = 90, 160, 12
BG = (0.95, 0.82, 0.12)                  # yellow
INK = (0.10, 0.16, 0.42)                 # navy
GHOSTS = ((1.0, 1.0, 1.0), (0.56, 0.71, 0.94))
FIG = (slice(25, 70), slice(65, 95))     # rows, columns of the figure


def frames():
    bare = np.broadcast_to(np.asarray(BG, np.float32), (H, W, 3)).copy()
    main = bare.copy()
    main[FIG] = INK
    return main, bare


def run(k, seed=1, n=N):
    main, bare = frames()
    return GL.frame(main, bare, k, n, seed=seed, shift=0.06, split=4.0, colors=GHOSTS)


def changed(img):
    return np.abs(img - frames()[1]).max(axis=2) > 1e-6


def test_the_figure_is_where_the_frame_differs_from_the_bare_one_and_render_noise_is_not():
    main, bare = frames()
    main[FIG[0], 64] = 0.5 * (np.asarray(BG) + np.asarray(INK))           # an antialiased column at its edge: all figure,
    main[5, 5] = np.asarray(BG) + 0.01                                     # so no halo stays behind; render noise: none
    main[6, 6] = np.asarray(BG) - 0.07                                     # a faint shade (a soft shadow's rim): partly
    a = GL.coverage(main, bare)
    assert a[FIG].min() == 1.0 and a[40, 64] == 1.0 and a[5, 5] == 0.0 and 0.0 < a[6, 6] < 1.0 and a[40, 120] == 0.0


def test_the_last_frame_is_the_bare_frame_and_the_frames_before_it_break_the_figure_up():
    main, bare = frames()
    assert np.array_equal(run(N - 1), bare)
    broken = [k for k in range(N - 1) if not np.array_equal(run(k), main) and not np.array_equal(run(k), bare)]
    assert len(broken) >= N // 2                                            # most frames glitch (blinks and stutters aside)
    outside = lambda c: c[:, :FIG[1].start - 6].any() or c[:, FIG[1].stop + 6:].any()   # noqa: E731
    assert any(outside(changed(run(k))) for k in broken)                    # slices jump out of the figure's columns


def test_the_ghosts_trail_the_figure_to_either_side_each_in_its_own_colour():
    main, bare = frames()
    seen = set()
    for k in range(N - 1):                                                  # no jumps: the slices stay in the figure's columns
        img = GL.frame(main, bare, k, N, seed=2, shift=0.0, split=4.0, colors=GHOSTS)
        for side, col_x, colour in (("left", FIG[1].start - 2, GHOSTS[0]), ("right", FIG[1].stop + 1, GHOSTS[1])):
            want = (1.0 - GL.GHOST) * np.asarray(BG) + GL.GHOST * np.asarray(colour)
            if (np.abs(img[FIG[0], col_x] - want).max(axis=1) < 0.02).any():
                seen.add(side)
    assert seen == {"left", "right"}


def test_the_picture_away_from_the_figure_is_never_touched():
    reach = int(round(0.06 * W)) + int(round(4.0 * 1.5)) + 1               # the farthest jump plus the ghosts' offset
    for seed in range(4):
        for k in range(N):
            c = changed(run(k, seed))
            assert not c[:FIG[0].start].any() and not c[FIG[0].stop:].any()           # rows without the figure
            assert not c[:, :FIG[1].start - reach].any() and not c[:, FIG[1].stop + reach:].any()


def test_a_glitch_is_repeatable_and_its_seed_changes_it():
    assert np.array_equal(run(4, seed=7), run(4, seed=7))
    assert any(not np.array_equal(run(k, seed=7), run(k, seed=8)) for k in range(N - 1))


def test_the_figure_thins_out_over_the_window():
    first, late = [], []
    for seed in range(24):
        first += [int((GL.coverage(run(k, seed), frames()[1]) > 0.5).sum()) for k in range(0, 3)]
        late += [int((GL.coverage(run(k, seed), frames()[1]) > 0.5).sum()) for k in range(N - 4, N - 1)]
    assert np.mean(late) < 0.5 * np.mean(first)


def test_a_frame_without_a_figure_is_left_as_it_is():
    main, bare = frames()
    assert np.array_equal(GL.frame(bare, bare, 3, N), bare)
