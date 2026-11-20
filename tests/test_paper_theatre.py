"""The paper theatre set (mkmmd.core.paper_theatre and its meshes, mkmmd.blender.library.sets.paper_theatre_parts, no
bpy): the options, the keep-out round the marionette, the drop-in and the yo-yo arrivals, the pendulums, the stop-motion
holds, the backdrop bent without stretching, the curtains' folds, closed plates and the card."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import parent_cutout_layout as PCL
from mkmmd.blender.library.sets import paper_theatre_parts as P
from mkmmd.core import palette as PAL
from mkmmd.core import paper_theatre as PT

DAWN = PAL.PALETTES["rose-pine-dawn"]
SIZES = [(2.2, 3.2, 2.35), (1.8, 2.8, 2.0), (3.2, 3.8, 4.0)]       # the defaults and the ends of the ranges


def _opts(**kw):
    return PT.options(dict({"name": "t", "kind": "paper_theatre", "at": [60.0, 0.0, 0.0]}, **kw))


def test_options_check_every_key():
    o = _opts()
    assert o["arrive"] is None and o["step"] == 0.0 and o["colors"] == {}
    with pytest.raises(PT.TheatreError) as e:
        _opts(arive=13.6)
    assert "arive" in str(e.value) and "arrive" in str(e.value) and "stagger" in str(e.value)    # lists the known
    for bad in ({"swing": "3"}, {"step": -1}, {"width": 9.0}, {"seed": 1.5}, {"arrive": True}, {"colors": {"sea": "pine"}}):
        with pytest.raises(PT.TheatreError):
            _opts(**bad)
    with pytest.raises(PT.TheatreError):
        PT.colours(_opts(colors={"sky": "chartreuse"}), DAWN)
    c = PT.colours(_opts(colors={"moon": "rose", "sky": "#ffffff"}), DAWN)
    assert set(c) == set(PT.ROLES) and c["sky"] == (1.0, 1.0, 1.0)
    assert c["moon"] == pytest.approx(PAL.srgb(DAWN["rose"]))


@pytest.mark.parametrize("size", SIZES)
def test_nothing_hangs_where_the_marionette_and_her_strings_are(size):
    """Every hanging piece and its thread stays out of |x| < 0.85, |y| < 0.6 through the whole motion: the drop, the
    yo-yo, the widest swing and twist the options allow."""
    lay = PT.layout(*size)
    o = _opts(arrive=2.0, swing=PT.RANGES["swing"][1], twist=PT.RANGES["twist"][1], width=size[0], height=size[1],
              depth=size[2])
    t = np.arange(0.0, 12.0, 1.0 / 60.0)
    ch, _hidden = PT.motion(o, lay, t)
    val = {(c.object, c.path, c.index): c.values for c in ch}
    kx, ky = PT.KEEP_OUT
    for h in lay.hangs:
        ax, ay, _az = h.anchor
        outline = np.concatenate([p.outline for p in h.part.plates])
        rx = np.abs(outline[:, 0]).max()                      # the piece turns about its thread: its reach
        swing_x = val[(h.name + "_hang", "rotation_euler", 1)]
        swing_y = val[(h.name + "_hang", "rotation_euler", 0)]
        length = -val[(h.name, "location", 2)]
        tw = np.abs(val[(h.name, "rotation_euler", 2)]).max()
        reach_y = rx * math.sin(tw) + 0.02                   # twisting brings its sides forward and back
        reach = (length + h.size[1]) * np.sin(np.abs(np.stack([swing_x, swing_y])).max(1))[:, None]
        x_near = np.abs(ax) - rx - reach[0].max()
        y_near = abs(ay) - reach_y - reach[1].max()
        assert x_near >= kx or y_near >= ky, (h.name, x_near, y_near)


@pytest.mark.parametrize("size", SIZES)
def test_the_pieces_hang_inside_the_backdrop_above_the_hills_and_clear_of_each_other(size):
    lay = PT.layout(*size)
    for h in lay.hangs:
        x, y, za = h.anchor
        outline = np.concatenate([p.outline for p in h.part.plates])
        lo, hi = outline.min(0), outline.max(0)
        assert abs(x) + max(abs(lo[0]), abs(hi[0])) + 0.12 < PT.sky_side(y, lay.sky), h.name
        assert za - h.length + lo[1] > 1.2, h.name                  # its bottom over the hills (they stop under 1 m)
        assert PT.KEEP_OUT[1] < y < lay.depth - 0.1
    for a in lay.hangs:
        for b in lay.hangs:
            if a.name < b.name and abs(a.anchor[1] - b.anchor[1]) < 0.1:      # on one batten: their sweeps never meet
                wa = 0.5 * a.size[0] + 0.12
                wb = 0.5 * b.size[0] + 0.12
                za = (a.anchor[2] - a.length - a.size[1], a.anchor[2] - a.length)
                zb = (b.anchor[2] - b.length - b.size[1], b.anchor[2] - b.length)
                overlap_z = za[0] < zb[1] and zb[0] < za[1]
                assert not overlap_z or abs(a.anchor[0] - b.anchor[0]) > wa + wb, (a.name, b.name)


def test_the_frame_drops_from_out_of_view_and_settles_within_a_second_and_a_half():
    lay = PT.layout()
    o = _opts(arrive=13.64)
    t = np.arange(10.0, 18.0, 1.0 / 60.0)
    ch, hidden = PT.motion(o, lay, t)
    val = {(c.object, c.path, c.index): c.values for c in ch}
    z = val[("frame", "location", 2)]
    before = t < 13.64
    assert (z[before] == lay.height).all() and hidden["proscenium"][before].all()
    assert not hidden["proscenium"][~before].any()
    # the front medium-wide (camera 4.6 m out at 1 m, 42 mm on a 9:16 frame) sees up to 2.45 m at the proscenium:
    # the foot of the frame is over that until the drop
    assert lay.height > 2.45
    land = 13.64 + PT.DROP_FALL
    after = t > land + 1.5
    assert np.abs(z[after]).max() < 1e-6
    assert np.degrees(np.abs(val[("frame", "rotation_euler", 0)][after])).max() < 0.1
    assert np.abs(val[("proscenium", "shape:flex", 0)][after]).max() < 0.05
    for c in lay.curtains:
        for i in (0, 1):
            assert np.degrees(np.abs(val[(c.name, "rotation_euler", i)][after])).max() < 0.2
        assert np.abs(val[(c.name, "shape:flex", 0)][after]).max() < 0.05
    falling = (t >= 13.64) & (t < land)
    assert (np.diff(z[falling]) < 0).all()                       # it falls faster and faster until it lands
    assert z[falling][0] > 0.8 * lay.height


def test_the_pieces_drop_in_on_their_threads_and_bounce_like_yoyos():
    lay = PT.layout()
    o = _opts(arrive=13.64)
    t = np.arange(13.0, 20.0, 1.0 / 240.0)
    ch, hidden = PT.motion(o, lay, t)
    val = {(c.object, c.path, c.index): c.values for c in ch}
    drops = []
    for h in lay.hangs:
        t0 = PT.release(o, h)
        drops.append(t0)
        length = -val[(h.name, "location", 2)]
        assert np.allclose(val[(h.name + "_thread", "scale", 2)], length)       # the thread reaches the piece
        assert hidden[h.name][t < t0].all() and not hidden[h.name][t >= t0].any()
        assert length[t < t0].max() < 0.05                                      # wound up on its batten until then
        caught = t >= t0
        L = length[caught]
        first = np.argmax(L >= h.length)
        assert t[caught][first] - t0 < 0.6                                       # a short drop
        stretch = L - h.length
        dips = np.flatnonzero((stretch[1:-1] > stretch[:-2]) & (stretch[1:-1] >= stretch[2:]) & (stretch[1:-1] > 0.01))
        assert 2 <= len(dips) <= 6, h.name                                       # two or three bounces, damped
        peaks = stretch[1:-1][dips]
        assert (np.diff(peaks) < 0).all()
        assert np.abs(stretch[t[caught] > t0 + 3.0]).max() < 0.005               # then it hangs still on its thread
    assert len({round(d, 3) for d in drops}) == len(drops)                     # staggered
    assert max(drops) - min(drops) == pytest.approx((len(drops) - 1) * o["stagger"])


def test_each_piece_swings_as_a_pendulum_of_its_own_length():
    lay = PT.layout()
    o = _opts()                                                                   # no arrive: hanging all along
    t = np.arange(0.0, 40.0, 1.0 / 60.0)
    ch, hidden = PT.motion(o, lay, t)
    assert not hidden
    val = {(c.object, c.path, c.index): c.values for c in ch}
    periods = []
    for h in lay.hangs:
        a = val[(h.name + "_hang", "rotation_euler", 1)]
        assert np.degrees(np.abs(a).max()) == pytest.approx(o["swing"] * h.swing, rel=0.02)
        up = np.flatnonzero((a[:-1] < 0) & (a[1:] >= 0))                         # upward zero crossings
        T = float(np.diff(t[up]).mean())
        assert T == pytest.approx(2 * math.pi * math.sqrt((h.length + 0.5 * h.size[1]) / PT.G), rel=0.01)
        periods.append(T)
    assert len({round(p, 2) for p in periods}) >= 7                           # their own lengths, not one beat


def test_the_sun_turns_at_its_rpm():
    lay = PT.layout()
    t = np.array([0.0, 30.0])
    ch, _ = PT.motion(_opts(sun_rpm=2.0), lay, t)
    rays = next(c for c in ch if c.object == "sun_rays")
    assert rays.path == "rotation_euler" and rays.index == 1
    assert rays.values[1] - rays.values[0] == pytest.approx(2.0 * math.pi)       # a turn in 30 s at 2 rpm


def test_step_holds_the_motion_like_stop_motion_on_the_stepped_grid():
    keys, times, interp = PT.key_times(1, 400, 181, 60.0, 12.0, 0.0)
    assert interp == "CONSTANT"
    assert np.allclose(np.diff(keys), 5.0)                                       # a pose every 5 frames at 60 fps
    assert keys[0] == 0.5 and ((keys[1:] + 0.5 - 181) % 5 == 0).all()           # half a frame before each step
    assert np.allclose(times[1:], (keys[1:] + 0.5 - 181) / 60.0)
    keys, times, _ = PT.key_times(1, 1743, 181, 60.0, 12.0, 12.35)             # aligned to a [[stepped]] window
    step_frames = keys[1:] + 0.5
    assert 181 + round(12.35 * 60) in step_frames
    keys, times, interp = PT.key_times(1, 400, 181, 60.0, 0.0)
    assert interp == "LINEAR" and len(keys) == 400 and times[180] == 0.0
    with pytest.raises(PT.TheatreError):
        PT.key_times(1, 400, 181, 30.0, 60.0)


@pytest.mark.parametrize("size", SIZES)
def test_the_backdrop_bends_without_stretching_and_the_sun_sits_on_its_flat_middle(size):
    lay = PT.layout(*size)
    a, R, D, yf, S = lay.sky
    s = np.linspace(-S, S, 4001)
    V = PT.sky_bend(np.stack([s, np.zeros_like(s), np.zeros_like(s)], 1), lay.sky)
    assert np.linalg.norm(np.diff(V, axis=0), axis=1).sum() == pytest.approx(2 * S, rel=1e-4)
    assert V[:, 1].min() == pytest.approx(yf) and V[:, 1].max() == pytest.approx(D)
    assert np.abs(V[:, 0]).max() == pytest.approx(a + R)
    rays = next(p for p in lay.statics if p.name == "sun_rays").plates[0].outline
    assert np.abs(lay.sun[0]) + np.abs(rays[:, 0]).max() < a + 0.03
    # the high pulled-back view (6.5 m out at 2.4 m, 32-35 mm) sees the backdrop's sides only behind the pillars
    outer = 0.5 * size[0] + PT.PILLAR
    assert (a + R + PT.SKY_CARD) * (PT.YP + 6.5) / (yf + 6.5) < outer


def test_the_curtains_keep_their_paper_width_and_their_folds_deepen_where_they_are_gathered():
    lay = PT.layout()
    for c in lay.curtains:
        u = np.linspace(0.0, 1.0, 2001)
        depths = []
        for z in (0.2, c.tie_z, c.inner_z.max() - 0.05):
            x, y = PT.pleat(c, u, np.full_like(u, z))
            assert np.hypot(np.diff(x), np.diff(y)).sum() == pytest.approx(c.paper, rel=0.02)
            depths.append(np.ptp(y))
        assert depths[1] > depths[0] and depths[1] > depths[2]
        assert np.sign(c.inner_x).min() == np.sign(c.inner_x).max() == c.side   # each on its own side
        assert np.abs(c.inner_x).min() > PT.KEEP_OUT[0] - 0.4


def test_outlines_never_cross_themselves():
    lay = PT.layout()
    plates = [p for part in lay.statics for p in part.plates] + list(lay.frame[0].plates)
    plates += [p for h in lay.hangs for p in h.part.plates] + [c.part for c in lay.curtains]
    for p in plates:
        assert PCL.crossings(p.outline) == [], p.role
        assert PT.polygon_area(p.outline) > 0
        for hole in p.holes:
            assert PT.inside(hole, p.outline).all() and PT.edge_distance(hole, p.outline).min() > 0.02


def test_every_plate_is_a_closed_outward_solid():
    """Cut, filled and rolled: an open plate shows its inside (or nothing) in a render."""
    lay = PT.layout()
    plates = [p for part in lay.statics for p in part.plates] + list(lay.frame[0].plates)
    plates += [p for h in lay.hangs for p in h.part.plates] + [c.part for c in lay.curtains]
    for k, p in enumerate(plates):
        m = P.plate_mesh(p, 100 + k)
        assert m.is_closed() and m.volume() > 0, (k, p.role)
    for c in lay.curtains:
        m, flex = P.curtain_mesh(c, 5)
        assert m.is_closed() and m.volume() > 0 and flex.shape == m.V.shape


def test_the_card():
    lay = PT.layout()
    card = PT.card("theatre", _opts(arrive=13.64), lay)
    names = [u["name"] for u in card["use"]["look"]]
    assert names[:2] == ["stage", "sun"] and "moon" in names
    assert {f"cloud{i}" for i in (1, 2, 3)} | {f"star{i}" for i in range(1, 6)} <= set(names)
    assert card["colliders"] == [{"type": "floor", "z": 0.0, "tag": "theatre"}] and card["lights"] == []
    assert set(card["materials"]) == {f"theatre_{r}" for r in PT.ROLES}
    assert card["theatre"]["lands"] == pytest.approx(13.64 + PT.DROP_FALL)
    assert len(set(card["objects"])) == len(card["objects"])
