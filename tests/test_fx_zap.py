"""Zaps (mkmmd.core.zap: the entry checked and timed; mkmmd.zap: a sheet's live sticker torn, somewhere else on the
time, a ghost left where it was)."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 120, 200
SKY = (0.55, 0.62, 0.7)
HER = (0.95, 0.2, 0.35)                     # her plate's colour: nothing else in the picture comes near it
T = 1.5                                     # the zap: on frame F0 + 45


def figure(g):
    m = np.zeros((H, W), np.float32)
    m[60:140, 40:80] = 1.0                  # 80 px tall, 40 wide: in her plate
    return m


def sky(g):
    return np.full((H, W, 3), SKY, np.float32)


def clouds(g):
    """A ground that drifts 2 px right a frame: soft bands, so its motion can be followed."""
    x = np.arange(W, dtype=np.float32) - 2.0 * g
    v = 0.5 + 0.25 * np.sin(x / 7.0)[None, :] * np.cos(np.arange(H, dtype=np.float32) / 11.0)[:, None]
    return np.dstack([v * 0.9, v, v * 1.1]).astype(np.float32)


def frames(with_zap, ground=sky, **zap):
    her = {"kind": "sheet", "name": "her", "from": 0.5, "to": 3.5, "objects": "Rin*", "shot": "plate", "paper": False,
           "drift": False, "edge": False, "shadow": False, "at": [0.3, 0.45], "height": 0.3,
           "keys": [{"t": 0.5}, {"t": T, "dx": 0.3, "dy": 0.4, "ease": "step"}]}     # 0.3 frame heights tall
    fx = [{"kind": "zap", "from": 1.0, "to": 3.5, "sheet": "her", "zaps": [{"t": T}], **zap}, her] if with_zap else [her]
    data = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}, {"name": "plate", "plate": True}], "fx": fx}
    pl = TR.plan(data, FPS, F0, DAWN)
    return pl, FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=ground,
                          point_of=lambda key, g: ((W / 2, H / 2), None), figure_of=lambda key, g: figure(g),
                          plate_of=lambda shot, g: np.full((H, W, 3), HER, np.float32))


def drawn(ffx, g):
    return ffx.frame(g, ffx._frame_of(g))


def hers(pic):
    return np.abs(pic - np.asarray(HER, np.float32)).max(2) < 0.04


def test_on_its_time_she_is_whole_at_the_new_place_her_ghost_where_she_was_and_she_is_untorn_by_the_snap():
    pl, ffx = frames(True)
    _, plain = frames(False)
    z = pl["fx"][0]["zaps"][0]
    f = z["frame"]
    assert f == F0 + 45 and (z["lead"], z["snap"]) == (3, 2)
    seq = {g: drawn(ffx, g) for g in range(f - 4, f + 4)}          # in order, as mk post draws them
    was, now = hers(plain.frame(f - 1, np.full((H, W, 3), SKY, np.float32))), hers(drawn(plain, f))
    assert was.sum() > 1500 and now.sum() > 1500 and not (was & now).any()      # two places, apart
    assert (hers(seq[f]) & now).sum() >= 0.85 * now.sum()          # on the time: there, all but a little of the tear
    assert not (hers(seq[f]) & was).any()                           # and nothing of her picture left where she was ...
    ghost = np.abs(seq[f] - np.asarray(SKY, np.float32)).max(2) > 0.05
    assert (ghost & was).sum() > 0.5 * was.sum()                    # ... but her ghost
    torn = np.abs(seq[f - 1] - np.asarray(SKY, np.float32)).max(2) > 0.05
    assert (torn & ~was).sum() > 0.1 * was.sum()                    # the frame before: her slices slid apart
    assert np.array_equal(hers(seq[f + 2]), now)                    # two frames on (the snap): whole again


def test_the_ghost_is_gone_by_its_life_and_the_picture_is_as_without_the_zap():
    pl, ffx = frames(True, ghost={"life": 0.5})
    _, plain = frames(False)
    z = pl["fx"][0]["zaps"][0]
    end = z["frame"] + z["ghost"]                                   # 15 frames after the zap: gone on this one
    pics = {g: drawn(ffx, g) for g in range(z["frame"] - 4, end + 2)}
    assert not np.array_equal(pics[end - 1], drawn(plain, end - 1))   # still there, faint, on its last frame
    for g in (end, end + 1):
        assert np.array_equal(pics[g], drawn(plain, g))


def test_the_ghost_is_dragged_along_the_picture_under_it_and_drawn_out_of_order_as_in_order():
    pl, ffx = frames(True, clouds)
    z = pl["fx"][0]["zaps"][0]
    g = z["frame"] + 6
    ahead = {k: drawn(ffx, k) for k in range(z["frame"] - 4, g + 1)}
    _, fresh = frames(True, clouds)                                 # a worker starting on this frame
    assert np.array_equal(drawn(fresh, g), ahead[g])
    assert not FF.ZPF.ZP.safe(pl["fx"][0], g) and FF.ZPF.ZP.safe(pl["fx"][0], z["frame"])
    _, still = frames(True)                                         # the same ghost over a still sky
    seen = {k: drawn(still, k) for k in range(z["frame"] - 4, g + 1)}

    def cols(pic, ground):                                          # where the ghost is, above her new place
        return np.nonzero((np.abs(pic - ground).max(2) > 0.08)[:100, :75].any(0))[0]
    assert cols(ahead[g], clouds(g)).mean() > cols(seen[g], sky(g)).mean() + 6     # 2 px a frame right, 6 frames on


def test_a_sheet_lays_its_sticker_as_it_did_before_zaps_where_none_names_it():
    class Before(FF.FrameFx):
        def _lay_live(self, fx, frame, out, face, ring, shadow):      # the sheets' own code before the hook
            if shadow is not None and shadow[3] > 0:
                out = FF._cast(out, ring, shadow)
            return face + out * (1.0 - ring[..., None])
    her = {"kind": "sheet", "from": 0.5, "to": 3.5, "objects": "Rin*", "at": [0.4, 0.6], "height": 0.4,
           "keys": [{"t": 0.5}, {"t": 2.0, "dx": 0.1, "turn": 20.0}], "slap": {"t": 0.6}}
    data = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": [her]}
    pl = TR.plan(data, FPS, F0, DAWN)
    kw = dict(fps=FPS, frame0=F0, frame_of=lambda g: np.full((H, W, 3), SKY, np.float32),
              point_of=lambda key, g: ((W / 2, H / 2), None), figure_of=lambda key, g: figure(g))
    a, b = FF.FrameFx(pl["fx"], (W, H), **kw), Before(pl["fx"], (W, H), **kw)
    for g in (F0 + 18, F0 + 30, F0 + 60):
        assert np.array_equal(drawn(a, g), drawn(b, g))


def test_a_zap_that_goes_nowhere_or_outlives_its_span_is_refused():
    with pytest.raises(TR.TransitionError, match="same place"):
        _zap_plan({"t": 2.0})                                       # her keys do not step on 2.0
    with pytest.raises(TR.TransitionError, match="outside the span"):
        _zap_plan({"t": T, "life": 2.5})


def _zap_plan(z):
    her = {"kind": "sheet", "name": "her", "from": 0.5, "to": 3.5, "objects": "Rin*", "paper": False,
           "keys": [{"t": 0.5}, {"t": T, "dx": 0.3, "ease": "step"}]}
    data = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}],
            "fx": [{"kind": "zap", "from": 1.0, "to": 3.5, "sheet": "her", "zaps": [z]}, her]}
    return TR.plan(data, FPS, F0, DAWN)


def test_a_steps_key_lands_on_the_frame_of_its_second():
    keys = FX._moves([{"t": 0.0}, {"t": 1.51, "dx": 0.2, "ease": "step"}], "k")
    half = 0.5 / FPS
    assert FX.move_at(keys, 1.5, half)["dx"] == 0.2                 # frame 45 (1.5 s) is the frame of 1.51
    assert FX.move_at(keys, 1.5 - 1.0 / FPS, half)["dx"] == 0.0     # the frame before holds
