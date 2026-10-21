"""The space kit (mkmmd.core.space; the builders are mkmmd/blender/library/props/space.py): the line's segments, its
retraction and ripples, the round board's cells and its Kepler shear, the chess pieces, the stars, the floors' patterns, the
starfield's leapfrogging tiles, the disc's orbits; then a real build of them (skipped without Blender)."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN
from mkmmd.core import space as SP


def _closed(mesh):
    """Every edge of the mesh is shared by exactly two faces (a closed surface)."""
    faces = [list(q) for q in mesh.Q] + [list(t) for t in mesh.T]
    count = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            k = (min(a, b), max(a, b))
            count[k] = count.get(k, 0) + 1
    return set(count.values()) == {2}


# ---------------------------------------------------------------------------------------------------------------- line
@pytest.mark.parametrize("anchor", ["end", "start"])
def test_the_segments_cover_the_line_from_the_anchor(anchor):
    segs = SP.segments(5.1, 1.0, anchor)
    assert len(segs) == 6 and [a for _, _, a in segs] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    spans = sorted((x0, x1) for x0, x1, _ in segs)
    assert spans[0][0] == pytest.approx(0.0) and spans[-1][1] == pytest.approx(5.1)
    assert all(b[0] == pytest.approx(a[1]) for a, b in zip(spans, spans[1:]))
    first = segs[0]
    assert (first[1] == pytest.approx(5.1)) if anchor == "end" else (first[0] == 0.0)    # the first touches the anchor


def test_the_line_retracts_into_its_anchor_and_the_driver_says_the_same_as_numpy():
    L, s = 4.0, 1.0
    segs = SP.segments(L, s, "end")
    for e in (1.0, 0.6, 0.3, 0.0):
        shown = [float(SP.segment_scale(e, a, s, L)) for _, _, a in segs]
        assert sum(shown) * s == pytest.approx(max(e * L, len(segs) * SP.MIN_SCALE * s), abs=1e-3)
        assert shown == sorted(shown, reverse=True)                     # the one nearest the anchor goes last
        for (_, _, a), want in zip(segs, shown):
            got = eval(SP.segment_expr(a, s, L), {"__builtins__": {}},                     # noqa: S307
                       {"clamp": lambda x, lo, hi: max(lo, min(hi, x)), "e": e})
            assert got == pytest.approx(want)


def test_ticks_every_spacing_and_every_bar_th_long():
    t = SP.ticks(2.0, 0.5, 2)
    assert [x for x, _ in t] == [0.0, 0.5, 1.0, 1.5, 2.0] and [lg for _, lg in t] == [True, False, True, False, True]
    assert SP.ticks(2.0, 0.0, 4) == []


def test_a_ripple_runs_out_both_ways_fades_and_is_gone_after_its_life():
    xs = np.linspace(-6.0, 6.0, 1201)
    z = SP.ripple([0.0, 0.5, 1.0, 1.1], xs, [(0.0, 0.0, -0.05)], speed=7.0, decay=0.3, width=0.5, life=1.0)
    assert z[0, 600] == pytest.approx(-0.05)                            # a dip under the hit
    left, right = xs[np.argmin(z[1, :600])], xs[600 + np.argmin(z[1, 600:])]
    assert left == pytest.approx(-3.5, abs=0.05) and right == pytest.approx(3.5, abs=0.05)   # 7 m/s for 0.5 s
    assert np.abs(z[1]).max() == pytest.approx(0.05 * math.exp(-0.5 / 0.3), rel=1e-3)
    assert not z[3].any()


def test_a_bad_line_says_what_to_change():
    with pytest.raises(SP.SpaceError, match="options"):
        SP.line_options({"lenght": 3})
    with pytest.raises(SP.SpaceError, match="anchor"):
        SP.line_options({"anchor": "middle"})
    with pytest.raises(SP.SpaceError, match=r"ripples\[0\]"):
        SP.line_options({"ripples": [[1.0, 2.0]]})
    assert SP.line_options({"love": "#ff0000"})["length"] == 40.0          # a palette slot is a colour override


# ---------------------------------------------------------------------------------------------------------------- board
def test_cells_alternate_round_a_ring_and_from_ring_to_ring():
    o = SP.board_options({"rings": 4, "sectors": 16})
    for i in range(4):
        row = [SP.light(i, j) for j in range(16)]
        assert all(a != b for a, b in zip(row, row[1:] + row[:1]))
        if i:
            assert row != [SP.light(i - 1, j) for j in range(16)]
    m = SP.ring_mesh(o, 0)
    assert set(np.unique(m.Qm)) == {0, 1} and (m.Qm == 0).sum() == (m.Qm == 1).sum()
    with pytest.raises(SP.SpaceError, match="even"):
        SP.board_options({"sectors": 15})


def test_a_cell_is_a_closed_slab_inside_its_ring_and_sector():
    o = SP.board_options({"inner": 2.0, "width": 1.0, "sectors": 8, "gap": 0.1})
    c = SP.cell_mesh(2.0, 3.0, 0.0, math.pi / 4, 0.05, 0.1)
    assert _closed(c)
    r = np.hypot(c.V[:, 0], c.V[:, 1])
    a = np.arctan2(c.V[:, 1], c.V[:, 0])
    assert r.min() == pytest.approx(2.05) and r.max() == pytest.approx(2.95)
    assert a.min() > 0 and a.max() < math.pi / 4 and c.V[:, 2].max() == 0.0 and c.V[:, 2].min() == -0.05
    assert SP.cell_centre(o, 0, 0) == pytest.approx((2.5 * math.cos(math.pi / 8), 2.5 * math.sin(math.pi / 8)))


def test_inner_rings_orbit_faster_by_keplers_law():
    o = SP.board_options({"inner": 2.0, "width": 1.0, "rings": 3, "omega": 1.2})
    w = SP.omegas(o)
    assert w[0] == pytest.approx(1.2) and w[1] / w[0] == pytest.approx((3.5 / 2.5) ** -1.5)
    x0, y0 = SP.cell_centre(o, 2, 5)
    x1, y1 = SP.cell_centre(o, 2, 5, phase=1.0)
    assert math.atan2(y1, x1) - math.atan2(y0, x0) == pytest.approx(w[2])
    assert SP.board_options({"kepler": 0.0, "rings": 3}) and SP.omegas({**o, "kepler": 0.0}) == [1.2] * 3


# ---------------------------------------------------------------------------------------------------------------- chess
@pytest.mark.parametrize("piece", SP.PIECES)
def test_every_piece_stands_on_the_floor_and_is_its_height(piece):
    m = SP.piece_mesh(piece, 2.0)
    lo, hi = m.bbox()
    assert lo[2] == pytest.approx(0.0, abs=1e-9) and 1.7 < hi[2] <= 2.2
    assert abs(lo[0] + hi[0]) < 0.01 or piece == "knight"
    if piece == "knight":
        assert hi[0] > -lo[0] + 0.1                                     # the head faces +X
    with pytest.raises(SP.SpaceError, match="one of"):
        SP.chess_options({"piece": "dragon"})


# ---------------------------------------------------------------------------------------------------------------- stars
def test_a_star_has_its_points_and_its_flash_returns_to_rest():
    P = SP.star_outline(5, 0.4)
    r = np.hypot(P[:, 0], P[:, 1])
    assert len(P) == 10 and np.allclose(r[::2], 1.0) and np.allclose(r[1::2], 0.4) and P[0, 1] == pytest.approx(1.0)
    s = SP.flash(np.array([0.0, 0.9, 1.0, 1.2, 1.35, 2.0]), 1.0, 0.35, 1.8)
    assert s[0] == 1.0 and s[2] == pytest.approx(1.8) and 1.0 < s[3] < 1.8 and s[4] == pytest.approx(1.0) and s[5] == 1.0
    m = SP.star_mesh(4, 0.3, 0.1, 0.004)
    lo, hi = m.bbox()
    assert hi[1] == pytest.approx(0.0, abs=1e-9) and lo[1] == pytest.approx(-0.004)        # it faces -Y
    with pytest.raises(SP.SpaceError, match=r"points\[1\]"):
        SP.stars_options({"points": [[0, 0, 1], [1, 2]]})


# ---------------------------------------------------------------------------------------------------------------- floor
def test_a_checker_is_light_and_dark_near_and_its_own_grey_far_away():
    c = 2.0
    near = SP.soften(5.0)
    assert SP.checker(1.0, 1.0, c, near) == pytest.approx(1.0, abs=1e-3)       # the middle of a light square
    assert SP.checker(3.0, 1.0, c, near) == pytest.approx(0.0, abs=1e-3)       # its neighbour is dark
    far = SP.soften(5000.0)                                                    # a footprint many squares wide
    xs = np.linspace(0.0, 40.0, 4001)
    assert np.abs(SP.checker(xs, 0.37, c, far) - 0.5).max() < 0.05             # no moire: the board's grey


def test_a_grid_line_s_light_stays_line_over_cell_however_far_it_is():
    c, line = 2.0, 0.04
    xs = np.linspace(-1.0, 1.0, 20001)                                          # one cell, a line through its middle
    for dist in (2.0, 20.0, 200.0):
        cover = SP.grid_lines(xs, 0.7, c, line, SP.soften(dist))                # y = 0.7: away from the other lines
        assert cover.mean() * c == pytest.approx(line, rel=0.08)               # the light it sends, near or far
    assert SP.grid_lines(0.0, 0.7, c, line, SP.soften(2.0)) == pytest.approx(1.0) and \
        SP.grid_lines(0.5, 0.7, c, line, SP.soften(2.0)) == 0.0                 # sharp up close: on it, then off it


@pytest.mark.parametrize("slots, frag", [({"kind": "lava"}, "kind is one of"), ({"fade": [100, 50]}, "near < far"),
                                         ({"line": 3.0}, "thinner than its cell"), ({"rough": 2}, "rough is 0 .. 1")])
def test_a_bad_floor_says_what_to_change(slots, frag):
    with pytest.raises(SP.SpaceError, match=frag):
        SP.floor_options(slots)


# ---------------------------------------------------------------------------------------------------------------- starfield
def test_the_starfield_tiles_leapfrog_and_always_cover_twice_their_length():
    o = SP.starfield_options({"length": 10.0, "speed": 7.0, "surge": [[1.0, 0.0], [2.0, 30.0]]})
    ts = np.linspace(0.0, 5.0, 501)
    s = SP.travel(ts, o)
    assert s[0] == 0.0 and s[-1] == pytest.approx(7.0 * 5.0 + 30.0)            # the speed plus the surge
    assert np.all(np.diff(s) > 0)
    a, b = SP.tile_x(s, 0, 10.0), SP.tile_x(s, 1, 10.0)
    assert np.all((a >= -10.0) & (a < 10.0)) and np.all((b >= -10.0) & (b < 10.0))
    lo = np.minimum(a, b)
    assert np.allclose(np.abs(a - b), 10.0) and np.allclose(np.minimum(lo, -10.0 + np.mod(lo + 10.0, 10.0)), lo)
    pts, sizes = SP.star_points(SP.starfield_options({"count": 300, "clear": 4.0, "seed": 3}))
    assert len(pts) == 300 and np.hypot(pts[:, 1], pts[:, 2]).min() >= 4.0     # none where she walks
    assert np.array_equal(pts, SP.star_points(SP.starfield_options({"count": 300, "clear": 4.0, "seed": 3}))[0])
    assert 0.35 <= sizes.min() and sizes.max() <= 1.0


# ---------------------------------------------------------------------------------------------------------------- disc
def test_the_disc_s_inside_orbits_faster_by_kepler():
    o = SP.disc_options({"inner": 2.0, "outer": 8.0, "spin": 0.5})
    assert SP.disc_turn(o, 2.0, 1.0) == pytest.approx(math.pi)                  # half a turn a second at the rim
    assert SP.disc_turn(o, 8.0, 1.0) == pytest.approx(math.pi * 4.0 ** -1.5)
    m = SP.disc_mesh(o)
    r = np.hypot(m.V[:, 0], m.V[:, 1])
    assert r.min() == pytest.approx(2.0) and r.max() == pytest.approx(8.0) and np.all(m.V[:, 2] == 0.0)
    with pytest.raises(SP.SpaceError, match="outer must be more"):
        SP.disc_options({"inner": 5.0, "outer": 4.0})


def test_a_shard_is_a_thin_closed_sliver_facing_the_picture():
    from mkmmd.core import burst as BU
    m = BU.mesh("shard")
    lo, hi = m.bbox()
    assert _closed(m) and hi[0] - lo[0] == pytest.approx(1.0, abs=1e-6)        # unit width
    assert hi[1] - lo[1] == pytest.approx(0.02, abs=1e-6) and hi[2] - lo[2] < 0.35   # thin, and long rather than tall
    BU.options({"mix": {"shard": 1.0}})

# ---------------------------------------------------------------------------------------------------------------- build
def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


SCENE = """[project]
name = "t"
fps = 30
frame0 = 31
duration = 2.0
blend = "build/t.blend"

[[prop]]
name = "rope"
card = "library:line"
slots = { length = 4, seg = 1.0, ticks = 1.0, ripples = [[1.0, 1.5, -0.05]] }

[[prop]]
name = "bh"
card = "library:hole"
at = [6.0, 0.0, 0.0]
slots = { radius = 1.0 }

[[prop]]
name = "board"
card = "library:round_board"
at = [6.0, 0.0, 0.0]
slots = { inner = 1.2, width = 0.6, rings = 3, sectors = 8 }

[[prop]]
name = "n1"
card = "library:chess"
parent = "board_r2"
at = [2.1, 0.0, 0.0]
slots = { piece = "knight", height = 1.0 }

[[prop]]
name = "sky"
card = "library:stars"
at = [0.0, 2.0, 0.0]
slots = { points = [[0.0, 1.5, 1.0, 1.0], [1.0, 2.0, 1.0, 1.5]] }

[[prop]]
name = "ground"
card = "library:floor"
slots = { kind = "grid", cell = 2.0 }

[[prop]]
name = "warp"
card = "library:starfield"
slots = { count = 40, length = 20.0, speed = 6.0 }

[[prop]]
name = "acc"
card = "library:disc"
at = [6.0, 0.0, 0.0]
slots = { inner = 1.2, outer = 3.0 }

[audio]
hits = "audio/hits.json"

[[prop]]
name = "glass"
card = "library:burst"
slots = { count = 6, mix = { shard = 1.0 } }

[[key]]
target = "glass"
prop = "start"
lane = "kick"
min_db = -3

[[key]]
target = "rope"
prop = "extent"
keys = [[0.0, 1.0], [1.0, 0.5]]
interp = "LINEAR"

[[key]]
target = "board"
prop = "phase"
keys = [[0.0, 0.0], [1.0, 2.0]]
interp = "LINEAR"

[[key]]
target = "board_hot_r1s3"
prop = "hide_render"
keys = [[0.0, 1], [0.5, 0]]
interp = "CONSTANT"
"""


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_the_space_kit_builds_and_moves_as_its_maths_says(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(SCENE, encoding="utf-8")
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "hits.json").write_text(json.dumps({"kick": [{"t": 0.2, "db": 0.0}, {"t": 0.6, "db": -9.0},
                                                                        {"t": 0.9, "db": -1.0}]}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        MAIN.main(["build", "--project", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert e.value.code == 0, out.get("error")
    assert not [ln for ln in out.get("log", []) if "WARNING" in ln]
    expr = ("[obj('rope_seg0').matrix.to_scale()[0], obj('rope_seg3').matrix.to_scale()[0], obj('rope_seg2').loc[2], "
            "obj('board_r0').euler[2], obj('board_r2').euler[2], obj('board_hot_r1s3').visible, "
            "obj('sky_star0').matrix.to_scale()[0], obj('n1_piece').loc[0], obj('warp_tile0').loc[0], "
            "obj('warp_tile1').loc[0], bpy.context.scene.eevee.use_raytracing, bpy.data.objects['glass']['start']]")
    with pytest.raises(SystemExit):
        MAIN.main(["q", str(tmp_path / "build" / "t.blend"), expr, "--frames", "31,46,61", "--project", str(tmp_path)])
    rows = json.loads(capsys.readouterr().out)["values"]
    f31, f46, f61 = rows
    assert f31[0] == pytest.approx(1.0) and f31[1] == pytest.approx(1.0)                 # all there at first
    assert f61[0] == pytest.approx(1.0) and f61[1] < 0.01                                # half gone, the far end first
    assert f61[2] != 0.0 and f46[2] == 0.0 == f31[2]                                     # the ripple, from its hit on
    o = SP.board_options({"inner": 1.2, "width": 0.6, "rings": 3, "sectors": 8})
    w = SP.omegas(o)
    assert f61[3] == pytest.approx(math.degrees(2.0 * w[0]), abs=1e-3)                  # mk q's euler is degrees
    assert f61[4] == pytest.approx(math.degrees(2.0 * w[2]), abs=1e-3)
    assert f31[5] is False and f46[5] is True                                            # the attacked square shows
    assert f61[6] == pytest.approx(1.8, abs=0.02) and f31[6] == pytest.approx(1.0)      # the star flashes at 1 s
    x0 = SP.cell_centre(o, 2, 0)                                                         # the knight rides ring 2
    assert abs(f61[7] - (6.0 + 2.1 * math.cos(2.0 * w[2]))) < 1e-3 and x0
    so = SP.starfield_options({"count": 40, "length": 20.0, "speed": 6.0})
    for row, t in ((f31, 0.0), (f46, 0.5), (f61, 1.0)):                                  # the tiles stream at its speed
        s = SP.travel([t], so)
        assert row[8] == pytest.approx(float(SP.tile_x(s, 0, 20.0)[0]), abs=1e-4)
        assert row[9] == pytest.approx(float(SP.tile_x(s, 1, 20.0)[0]), abs=1e-4)
    assert f31[10] is True                                                               # the floor mirrors what is seen
    assert f31[11] == pytest.approx(0.2) and f46[11] == pytest.approx(0.2) and f61[11] == pytest.approx(0.9)   # loud kicks
