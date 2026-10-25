"""The grid collage's layouts and division (mkmmd.core.framefx: `cells` of [columns, rows], `divide`, grid_cells; mkmmd.framefx:
the cells drawn) and the silkscreen fill of every collage (flat inks by the figure's own luma, the key plate off register,
its line work)."""
import re

import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 64, 36
RED, BLUE, WHITE, BLACK, GREEN, YELLOW = ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0], [1.0, 1.0, 0.0])


def plan(fx):
    data = {"project": {"duration": 4.0}, "output": [{"name": "16x9", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


def grid(**kw):
    return {"kind": "grid", "from": 1.0, "to": 3.0, "objects": "Fig*", **kw}


def square(x0, x1, y0, y1, size=(W, H)):
    m = np.zeros((size[1], size[0]), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


@pytest.mark.parametrize("spec, frag", [
    (grid(cells=[0, 2]), "cells = 0: expected 1..16"),
    (grid(cells=[[2]]), "a layout is n (n x n) or [columns, rows]"),
    (grid(cells=[[2, 17]]), "cells = 17: expected 1..16"),
    (grid(cells="x"), "expected a whole number"),
    (grid(divide=-1), "divide = -1: expected at least 0"),
    (grid(fill="silkscreen", inks=[["#000000"]]), "ink sets of 2 to 4 colours"),
    (grid(fill="silkscreen", inks=[["#000000", "#111111", "#222222", "#333333", "#444444"]]), "2 to 4 colours"),
    (grid(fill="silk"), "colour 'silk'"),
    (grid(offset=4), "offset = 4: expected [x, y]"),
])
def test_bad_grid_and_silkscreen_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        plan([spec])


def test_cells_are_columns_and_rows_and_a_whole_number_is_a_square():
    norm = lambda **kw: FX.normalize(grid(**kw), DAWN)                          # noqa: E731
    assert norm(cells=3)["cells"] == [[3, 3]] and norm(cells=[[4, 2]])["cells"] == [[4, 2]]
    assert norm(cells=[1, [2, 1], [2, 2], [4, 2], [4, 4]])["cells"] == [[1, 1], [2, 1], [2, 2], [4, 2], [4, 4]]
    fx = dict(norm(cells=[[4, 2]], piece="text"), index=0)
    assert all(0 <= c < 4 and 0 <= r < 2 for c, r in (FX.piece_cell(fx, k) for k in range(12)))
    inks = norm(fill="silkscreen")["inks"]                                      # the palette's: the darker key first
    assert len(inks) == 4 and all(len(s) == 3 for s in inks)
    assert inks[0][0] == [round(c, 6) for c in PAL.srgb(DAWN["text"])]
    assert norm(fill=["silkscreen", "picture", "#ff0000"])["fill"] == ["silkscreen", None, RED]


def test_a_layout_of_columns_and_rows_shows_the_whole_frame_contain_fitted_in_each_cell():
    xx, yy = np.meshgrid(np.linspace(0, 1, W, dtype=np.float32), np.linspace(0, 1, H, dtype=np.float32))
    img = np.dstack([xx, yy, np.zeros_like(xx)])
    m = square(24, 40, 8, 28)
    out = FF.grid(img, m, (2, 1), lambda i, j: [BLUE, WHITE][i], lambda i, j: None)
    assert out[0:9, 0:32].reshape(-1, 3).tolist() == [BLUE] * 9 * 32            # above the half-size frame: its colour
    assert out[27:, 32:].reshape(-1, 3).tolist() == [WHITE] * 9 * 32            # below it in the other cell
    assert out[18, 16] == pytest.approx(img[18, 32], abs=0.03)                  # the picture at half size, centred
    assert out[18, 48] == pytest.approx(img[18, 32], abs=0.03)
    tall = FF.grid(img[:, :18], m[:, :18], (1, 2), lambda i, j: RED, lambda i, j: WHITE)   # 18 x 36: two 18 x 18 cells
    assert tall[0:4, :].reshape(-1, 3).tolist() == [RED] * 4 * 18 and tall[14:18, :].reshape(-1, 3).tolist() == [RED] * 72


def test_new_cells_grow_out_of_their_parents_eased_out_and_old_cells_fold_back_into_theirs():
    pl = plan([grid(cells=[1, [2, 1], [2, 2], [4, 2], [4, 4], [2, 2]], divide=0.2,
                    times=[1.2, 1.5, 1.8, 2.1, 2.4])])
    fx = pl["fx"][0]
    at = lambda f: FX.grid_cells(fx, F0 + f, fx["times"], FPS, F0)              # noqa: E731
    assert at(35) == [((0.0, 0.0, 1.0, 1.0), (0, 0))]
    assert at(36) == [((0.0, 0.0, 1.0, 1.0), (0, 0)), ((0.0, 0.0, 1.0, 1.0), (1, 0))]     # on the hit: the parent's place
    (b0, c0), (b1, c1) = at(39)                                                  # half way: 3/4 of the way, eased out
    assert b0 == pytest.approx((0.0, 0.0, 0.625, 1.0)) and b1 == pytest.approx((0.375, 0.0, 1.0, 1.0))
    assert at(42) == [((0.0, 0.0, 0.5, 1.0), (0, 0)), ((0.5, 0.0, 1.0, 1.0), (1, 0))]     # after `divide`: its own
    assert [len(at(f)) for f in (35, 44, 53, 62, 71)] == [1, 2, 4, 8, 16]              # 1 -> 2 -> 4 -> 8 -> 16
    split = {c: b for b, c in at(54)}
    assert split[(3, 1)] == pytest.approx(FX.box((2, 2), 1, 1)) and split[(1, 0)] == pytest.approx(FX.box((2, 2), 0, 0))
    four = {c: b for b, c in at(63)}                                             # [4, 2] -> [4, 4] on its hit
    assert four[(2, 3)] == pytest.approx(FX.box((4, 2), 2, 1)) and four[(0, 0)] == pytest.approx(FX.box((4, 2), 0, 0))
    fold = at(72)                                                                # [4, 4] -> [2, 2]: the old 16 cells
    assert len(fold) == 16 and fold[15] == ((0.75, 0.75, 1.0, 1.0), (1, 1))     # in the colours of their new cell
    assert dict((c, b) for b, c in at(80)) == {(i, j): FX.box((2, 2), i, j) for j in range(2) for i in range(2)}
    mid = at(75)
    assert mid[15][0] == pytest.approx(tuple(a + (b - a) * 0.75 for a, b in zip((0.75, 0.75, 1.0, 1.0),
                                                                                 FX.box((2, 2), 1, 1))))


def test_a_step_s_division_ends_before_the_next_hit_and_a_cut_has_none():
    pl = plan([grid(cells=[1, [2, 1], [2, 2]], divide=0.5, times=[1.2, 1.3]), grid(cells=[1, 2], times=[1.2])])
    fx, cut = pl["fx"]
    at = lambda fx, f: FX.grid_cells(fx, F0 + f, fx["times"], FPS, F0)          # noqa: E731
    (b0, _), _ = at(fx, 38)                                                      # 2 of the 3 frames to the next hit
    assert b0[2] == pytest.approx(1.0 - 0.5 * (1.0 - (1.0 - 2.0 / 3.0) ** 2))
    assert len(at(fx, 39)) == 4                                                  # the next step has begun
    assert at(cut, 36) == [(FX.box((2, 2), i, j), (i, j)) for j in range(2) for i in range(2)]


def test_a_dividing_grid_shows_both_copies_on_their_way_and_covers_the_frame():
    pl = plan([grid(cells=[1, [2, 1]], divide=0.2, times=[1.2], back=["#ff0000", "#0000ff"], fill="#ffffff")])
    run = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                     figure_of=lambda key, g: square(30, 34, 10, 26))
    img = np.zeros((H, W, 3), np.float32)
    out = run.frame(F0 + 39, img)                                               # boxes 0..40 and 24..64 px
    assert not (out.sum(-1) == 0).any()                                          # the frame covered by cells
    assert out[18, 4].tolist() == BLUE and out[18, 60].tolist() == RED
    assert out[18, 20].tolist() == WHITE and out[18, 44].tolist() == WHITE      # both copies, neither cut off
    assert out[18, 32].tolist() == RED                                           # between them: the later cell's ground


def test_a_silkscreen_prints_flat_inks_cut_at_the_figure_s_own_luma_quantiles():
    xx = np.tile(np.linspace(0.02, 0.1, 40, dtype=np.float32), (20, 1))       # a dark figure: its own quantiles
    pic = np.zeros((20, 60, 3), np.float32)
    pic[:, 10:50] = xx[..., None]
    m = square(10, 50, 0, 20, (60, 20))
    key, steps = FF.silk_bands(pic, m, 3, 0.0)
    inks = [BLACK, RED, GREEN]
    out = FF.silk_print(np.full((20, 60, 3), 0.5, np.float32), m, key, steps, inks, (0.0, 0.0))
    row = out[10, 10:50]
    kinds = [next((n for n, c in enumerate(inks) if np.allclose(p, c)), None) for p in row]
    flat = [k for k in kinds if k is not None]
    assert flat == sorted(flat) and len(flat) >= len(kinds) - 2                  # in order; a pixel of edge between
    assert [flat.count(k) for k in range(3)] == pytest.approx([8, 16, 16], abs=2)          # 20 %, 40 %, 40 %
    assert out[10, 5].tolist() == [0.5] * 3                                      # the ground untouched
    assert FF.silk_levels(xx, np.ones_like(xx), 4) == pytest.approx(np.quantile(xx, [0.2, 0.4667, 0.7333]) + 0.5 / 255,
                                                                    abs=1e-3)       # half a code value up: ties fall below


def test_the_key_plate_prints_off_register_and_on_the_ground_past_the_figure():
    pic = np.full((20, 60, 3), 0.8, np.float32)
    pic[:, 10:20] = 0.05                                                         # dark at its left and right ends
    pic[:, 44:50] = 0.05
    m = square(10, 50, 0, 20, (60, 20))
    key, steps = FF.silk_bands(pic, m, 2, 0.0)
    out = FF.silk_print(np.full((20, 60, 3), 0.5, np.float32), m, key, steps, [BLACK, RED], (4.0, 0.0))
    row = [p.tolist() for p in out[10]]
    assert row[11:14] == [RED] * 3                                               # where the key moved off: the ink
    assert row[15:23] == [BLACK] * 8 and row[24] == RED                          # (its edge pixels half inked)
    assert row[49:53] == [BLACK] * 4 and row[54] == [0.5] * 3                   # off the figure: on the ground


def test_the_key_plate_draws_the_lines_where_the_luma_dips_under_its_surroundings():
    pic = np.full((40, 40, 3), 0.8, np.float32)
    pic[:, :20] = 0.1                                                            # a dark half: the darkest band
    pic[18:20, 24:36] = 0.6                                                      # a faint line in the light half
    m = np.ones((40, 40), np.float32)
    plain, _ = FF.silk_bands(pic, m, 3, 0.0)
    lined, _ = FF.silk_bands(pic, m, 3, 0.0, line=3.0)
    assert plain[19, 30] == 0.0 and lined[19, 30] == 1.0                        # the line prints only with line work
    assert plain[10, 5] == 1.0 and lined[30, 30] == 0.0                         # the dark band, and the light stays


def test_a_silkscreen_grid_reads_its_inks_and_grounds_at_the_step_plus_the_cell():
    pl = plan([grid(cells=[[2, 1]], fill="silkscreen", times=[1.5], back=["#0000ff", "#ff0000"],
                    inks=[["#000000", "#00ff00", "#00ff00"], ["#000000", "#ffff00", "#ffff00"]], offset=[0, 0])])
    run = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                     figure_of=lambda key, g: square(16, 48, 6, 30))
    img = np.full((H, W, 3), 0.7, np.float32)
    img[6:12, 16:48] = 0.05                                                      # the top of the figure dark
    before, after = run.frame(F0 + 40, img), run.frame(F0 + 50, img)
    assert before[2, 16].tolist() == BLUE and before[2, 48].tolist() == RED     # grounds at step 0 + cell
    assert before[18, 16].tolist() == GREEN and before[18, 48].tolist() == YELLOW
    assert before[13, 16].tolist() == BLACK                                      # the key plate (the top, dark)
    assert after[2, 16].tolist() == RED and after[18, 16].tolist() == YELLOW     # one hit on: the next entries
