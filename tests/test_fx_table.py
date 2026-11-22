"""Tables (mkmmd.core.table: a sheet with a `clip`, its view, punches, touches, rings; mkmmd.table: the clip through its
grade and the view, the cup's rim cutting what floats, a tracked figure, the picture under the table in the cup) on a
small synthetic clip and figure."""
import math

import cv2
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import footage as FT
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import sheet as ST
from mkmmd.core import table as TB
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 90, 160                                  # a tall output
CW, CH = 160, 90                                # the clip: a cup of dark coffee in the middle of a lighter table
CUP = {"center": [0.5, 0.5], "radius": [0.3, 0.3]}


def cup_clip(path, n=12):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (CW, CH))
    assert vw.isOpened()
    for _ in range(n):
        f = np.full((CH, CW, 3), 170, np.uint8)
        cv2.circle(f, (CW // 2, CH // 2), int(0.3 * CH), 40, -1)
        vw.write(f)
    vw.release()
    return path


def table(**kw):
    return {"kind": "sheet", "from": 0.0, "to": 2.0, "clip": "clip.mp4", "drift": False, "cup": CUP, **kw}


def plan(fx, shots=None):
    data = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": shots or [{"name": "a", "from": 0.0, "to": 4.0}, {"name": "plate", "plate": True}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


def frames(tmp_path, fx, figure=None, point=None, under=None):
    cup_clip(tmp_path / "clip.mp4")
    pl = plan(fx)
    under = np.full((H, W, 3), (1.0, 0.0, 1.0), np.float32) if under is None else under
    plate = np.full((H, W, 3), (0.2, 0.9, 0.2), np.float32)
    return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: under,
                      point_of=point or (lambda key, g: ((W / 2, H / 2), None)),
                      figure_of=(lambda key, g: figure(g)) if figure else None, plate_of=lambda shot, g: plate,
                      footage_of=FT.source(tmp_path)), pl, under


def test_a_view_lands_on_its_keys_a_step_holds_to_its_second_and_a_push_carries_its_target_straight_in():
    view = TB._view(FX, [{"t": 0.0, "at": [0.3, 0.4], "height": 0.05}, {"t": 1.0, "at": [0.5, 0.5], "height": 1.0,
                                                                         "ease": "step"},
                         {"t": 3.0, "at": [0.3, 0.4], "height": 0.05, "ease": "inout"}], "v")
    size, tbl = (W, H), (CW, CH)
    key = [TB.similarity(TB.middle(size), *TB.key_map(k, size, tbl)) for k in view]
    assert np.allclose(TB.view_map(view, 0.999, size, tbl), key[0])          # held until the step's second ...
    assert np.allclose(TB.view_map(view, 1.0, size, tbl), key[1])            # ... and there on it, not a frame late:
    assert np.allclose(TB.view_map(view, 1.0 - 0.4 / FPS, size, tbl, 0.5 / FPS), key[1])   # on the frame of its second
    assert np.allclose(TB.view_map(view, 3.0, size, tbl), key[2])
    mid = TB.view_map(view, 2.0, size, tbl)                                  # inout: half way at the middle second
    assert TB.zoom_of(mid) == pytest.approx((TB.zoom_of(key[1]) + TB.zoom_of(key[2])) / 2.0)
    target = np.array([0.3 * CW - 0.5, 0.4 * CH - 0.5, 1.0])                 # the close key's middle slides straight in
    assert np.allclose(mid @ target, (key[1] @ target + np.asarray(TB.middle(size))) / 2.0)
    wide = {"height": 0.2, "width": 0.9}                                     # a width the tall frame must hold
    assert TB.frame_height(wide, 9 / 16) == pytest.approx(1.6) and TB.frame_height(wide, 16 / 9) == pytest.approx(0.5063,
                                                                                                                  abs=1e-3)


def test_a_punch_lands_with_the_cup_over_the_whole_frame_and_the_table_gives_way_to_the_cut_on_its_second():
    rest = TB.view_map(None, 0.0, (W, H), (CW, CH))
    p = {"t": 2.0, "dur": 0.3, "out": False, "fade": 0.35}
    M, k = TB.punch_map(p, 1.0, rest, (W, H), (CW, CH), CUP)
    inv = cv2.invertAffineTransform(M)
    c, r = np.array([0.5 * CW - 0.5, 0.5 * CH - 0.5]), 0.3 * CH
    for corner in ((-0.5, -0.5), (W - 0.5, -0.5), (-0.5, H - 0.5), (W - 0.5, H - 0.5)):   # the coffee covers the frame
        assert np.hypot(*(inv @ np.array([*corner, 1.0]) - c)) <= r * 1.001
    assert k == pytest.approx(1.0) and TB.inside_alpha(p, 0.9999) == 1.0
    out = {"t": 5.0, "dur": 0.3, "out": True, "fade": 0.35}
    assert not TB.gone([p], 1.99) and TB.gone([p], 2.0)                      # in: the cut's own frame from its second
    assert TB.gone([out], 4.99) and not TB.gone([out], 5.0)                  # out: the table from its second on
    assert TB.gone([p, out], 3.0) and not TB.gone([p, out], 5.2)


def test_the_clip_is_graded_through_the_map_and_the_cups_rim_cuts_a_pad_that_straddles_it(tmp_path):
    pad = {"at": [0.5 + 0.3 * CH / CW, 0.5], "size": 0.25, "color": "love", "light": "love", "veins": False, "edge": 0}
    fx = table(grade={"map": ["text", "gold"], "stretch": [1, 99]}, pads=[pad], view={"at": [0.5, 0.5], "height": 1.0,
                                                                                         "width": 1.8})
    ffx, pl, under = frames(tmp_path, [fx])
    out = ffx.frame(F0 + 3, under)
    text, gold, love = (np.asarray(FX._colour(c, DAWN, "c")) for c in ("text", "gold", "love"))
    M = TB.view_map(pl["fx"][0]["view"], 0.1, (W, H), (CW, CH))
    at = lambda x, y: out[tuple(np.rint(M @ np.array([x * CW - 0.5, y * CH - 0.5, 1.0])).astype(int))[::-1]]  # noqa: E731
    assert np.abs(at(0.5, 0.5) - text).max() < 0.08                         # the dark coffee: the map's dark end
    assert np.abs(at(0.05, 0.1) - gold).max() < 0.08                         # the light table: its light end
    edge = 0.5 + 0.3 * CH / CW
    assert np.abs(at(edge - 0.03, 0.5) - love).max() < 0.12                  # the pad on the coffee ...
    assert np.abs(at(edge + 0.04, 0.5) - gold).max() < 0.12                  # ... and none of it past the rim


def test_lumas_place_the_maps_colours_the_coffee_has_its_own_map_and_soften_calms_the_tables_speckle(tmp_path):
    vw = cv2.VideoWriter(str(tmp_path / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (CW, CH))
    ys, xs = np.mgrid[0:CH, 0:CW]
    f = np.where((xs // 4 + ys // 4) % 2 == 0, 110, 230).astype(np.uint8)      # a speckled table, luma 170 on average
    cv2.circle(f, (CW // 2, CH // 2), int(0.3 * CH), 40, -1)                     # the dark coffee
    for _ in range(12):
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()
    text, gold, love = (np.asarray(FX._colour(c, DAWN, "c")) for c in ("text", "gold", "love"))
    pl = plan([table(grade={"map": ["text", "gold"], "lumas": [0.55, 0.78], "soften": s}, view={"height": H / CH},
                     cup={**CUP, "map": ["love", "foam"], "lumas": [0.25, 0.6]}) for s in (0.0, 0.05)])
    under = np.zeros((H, W, 3), np.float32)
    outs = []
    for fx in pl["fx"]:
        ffx = FF.FrameFx([fx], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: under, point_of=lambda key, g: None,
                         footage_of=FT.source(tmp_path))
        outs.append(ffx.frame(F0 + 3, under))
    rough, calm = (o[36:45, 30:60] for o in outs)                              # the table above the cup: clip rows 1-9
    assert np.abs(outs[0][80, 45] - love).max() < 0.03 and np.abs(outs[1][80, 45] - love).max() < 0.03   # its own map
    assert np.abs(rough - text).max(axis=-1).min() < 0.08 and np.abs(rough - gold).max(axis=-1).min() < 0.08
    mid = text + (gold - text) * ((170 / 255 - 0.55) / (0.78 - 0.55))         # luma 170 between the two colours' lumas
    assert np.abs(calm - mid).max() < 0.06                                     # the speckle smoothed to its mean's colour
    with pytest.raises(FX.FxError, match="lumas"):
        FX.normalize(table(grade={"map": ["text", "gold"], "lumas": [0.2, 0.8], "stretch": [1, 99]}), DAWN)
    with pytest.raises(FX.FxError, match="rising"):
        FX.normalize(table(grade={"map": ["text", "gold"], "lumas": [0.8, 0.2]}), DAWN)


def test_a_table_holds_still_unless_given_a_drift_and_paper_still_drifts():
    bare = {k: v for k, v in table().items() if k != "drift"}
    assert FX.normalize(bare, DAWN)["drift"] is None                         # the loop's seam: no drift to differ by
    assert FX.normalize({**bare, "drift": 0.01}, DAWN)["drift"]["amount"] == 0.01
    assert FX.normalize({"kind": "sheet", "from": 0, "to": 2}, DAWN)["drift"] == ST.DRIFT


def test_drops_leave_from_the_rim_of_the_pad_she_lands_on_and_land_small_on_the_coffee_clear_of_the_pads(tmp_path):
    pads = [{"at": [0.5, 0.5], "size": 0.2}, {"at": [0.5 + 0.18 * CH / CW, 0.5], "size": 0.08, "seed": 2}]
    fx = table(pads=pads, touch={"t": 0.1}, drops={"count": 8, "size": 0.002, "reach": 0.25}, view={"height": 1.0})
    ffx, pl, _ = frames(tmp_path, [fx])
    st = ffx._table_state(pl["fx"][0])
    asp = CW / CH
    got = [(p, o) for p, o in zip(st["drops"]["places"], st["drops"]["origins"]) if p is not None]
    assert len(got) >= 6
    for p, o in got:
        assert math.hypot((p[0] - 0.5) * asp, p[1] - 0.5) < 0.3                  # on the coffee ...
        for q in pads:                                                            # ... off every pad
            assert math.hypot((p[0] - q["at"][0]) * asp, p[1] - q["at"][1]) > 0.5 * q["size"]
        assert math.hypot((o[0] - 0.5) * asp, o[1] - 0.5) == pytest.approx(0.1)   # thrown from her pad's rim ...
        assert (o[0] - 0.5) * (p[1] - 0.5) - (o[1] - 0.5) * (p[0] - 0.5) == pytest.approx(0.0, abs=1e-9)   # ... outward


def test_a_tracked_figure_lies_where_its_world_point_puts_it_however_the_plate_camera_zooms(tmp_path):
    def figure(g):                                    # a square round the point, half a metre across, in the plate
        p, m = point("x0t", g)
        half = 0.25 * m * H
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        return ((np.abs(xs - p[0]) < half) & (np.abs(ys - p[1]) < half)).astype(np.float32)

    def point(key, g):                                # the plate camera zooms in 2x from frame 10 and looks off centre
        return ((W / 2 - 0.5, H / 2 - 0.5), 0.25) if g < F0 + 10 else ((W / 2 + 7.5, H / 2 - 12.5), 0.5)

    fx = table(objects="Fig*", shot="plate", edge=False, shadow=False,
               track={"point": "(0, 0, 1)", "at": [0.4, 0.45], "metre": 0.2},
               view={"at": [0.4, 0.45], "height": 0.5})
    ffx, pl, under = frames(tmp_path, [fx], figure=figure, point=point)
    boxes = []
    for g in (F0 + 2, F0 + 20):
        out = ffx.frame(g, under)
        green = (out[..., 1] > 0.8) & (out[..., 0] < 0.3)
        ys, xs = np.nonzero(green)
        boxes.append((xs.min(), xs.max(), ys.min(), ys.max()))
    assert boxes[0] == pytest.approx(boxes[1], abs=1.5)                      # the same place and size on the table
    cx, cy = (boxes[0][0] + boxes[0][1]) / 2.0, (boxes[0][2] + boxes[0][3]) / 2.0
    assert (cx, cy) == pytest.approx(TB.middle((W, H)), abs=1.0)              # her point: the view's middle
    assert boxes[0][3] - boxes[0][2] + 1 == pytest.approx(0.5 * 0.2 / 0.5 * H, abs=2.0)   # half a metre: 0.1 table


def test_the_cuts_own_frame_lies_in_the_cup_untouched_inside_it_one_picture_with_the_wide_view(tmp_path):
    rng = np.random.default_rng(3)
    under = (rng.integers(0, 256, (H, W, 3)) / 255.0).astype(np.float32)        # the cut's frame: a shot from above
    metre, sh = 0.1, {"width": 0.03, "offset": [0.03, 0.03], "alpha": 0.5}

    def point(key, g):                     # the shot's lens follows the view: a metre is metre / its height in frames
        return TB.middle((W, H)), metre / (0.2 if g < F0 + 30 else 1.8)
    inside = {"point": "(0, 0, 0)", "at": [0.5, 0.5], "metre": metre, "shadow": sh}
    fx = table(inside=inside, view=[{"t": 0.0, "at": [0.5, 0.5], "height": 0.2},
                                    {"t": 1.0, "height": 1.8, "ease": "step"}])
    turned = table(inside=inside, view={"at": [0.5, 0.5], "height": 0.2, "turn": -90})
    for f in (fx, turned):
        ffx, pl, _ = frames(tmp_path, [f], point=point, under=under)
        assert np.array_equal(ffx.frame(F0 + 3, under), under)                  # inside the cup: the frame untouched
    ffx, pl, _ = frames(tmp_path, [fx], point=point, under=under)
    wide = ffx.frame(F0 + 33, under)
    mid = (slice(H // 2 - 6, H // 2 + 6), slice(W // 2 - 6, W // 2 + 6))
    assert np.array_equal(wide[mid], under[mid])                              # the same picture, at its own size
    M = TB.view_map(pl["fx"][0]["view"], 1.1, (W, H), (CW, CH))
    px = lambda x, y: tuple(np.rint(M @ np.array([x * CW - 0.5, y * CH - 0.5, 1.0])).astype(int))[::-1]  # noqa: E731
    surface, text = (np.asarray(FX._colour(c, DAWN, "c")) for c in ("surface", "text"))
    assert np.abs(wide[px(0.5, 0.08)] - surface).max() < 0.08                   # past the rim: the table, graded
    k = 0.27 / math.sqrt(2.0)                                                 # just inside the rim, on the lamp's side
    lit, far = px(0.5 - k * CH / CW, 0.5 - k), px(0.5 + k * CH / CW, 0.5 + k)  # ... and on the far side
    assert np.abs(wide[lit] - text).max() < np.abs(under[lit] - text).max() - 0.05   # in the wall's shadow
    assert np.array_equal(wide[far], under[far])


def test_a_slapped_doodle_is_hidden_until_its_drop_and_lands_on_its_second():
    d = {"angle": 0.0, "spin": 0.0, "twist": 1.0}
    slap = {"t": 2.0, **ST.DSLAP}
    at = lambda t: ST.doodle_at(t, d, (0.5, 0.5), 0.0, None, [], None, None, 0.0, None, slap)    # noqa: E731
    assert not at(2.0 - slap["drop"] - 0.01)["shown"]
    fall = at(2.0 - slap["drop"] / 2.0)
    assert fall["shown"] and fall["scale"] > 1.05 and fall["lift"] > 0.2      # held large, falling
    assert at(2.0)["lift"] == 0.0 and at(2.0 + slap["settle"] + 0.01)["scale"] == pytest.approx(1.0)
    fx = FX.normalize({"kind": "sheet", "from": 0, "to": 4, "doodles": {"count": 2, "slap": {"times": [2.31, 2.12]},
                                                                         "hop": {"lane": "notes"}}}, DAWN)
    assert fx["doodles"]["slap"]["times"] == [2.12, 2.31] and fx["doodles"]["hop"]["lane"] == "notes"
    assert ST.slap_order(fx["doodles"], [(0.1, 0.1), None, (0.2, 0.2)]) == {
        0: {"t": 2.12, **ST.DSLAP}, 2: {"t": 2.31, **ST.DSLAP}}


def test_a_hand_rides_a_table_point_carried_and_zoomed_by_the_view(tmp_path):
    view = [{"t": 0.0, "at": [0.5, 0.5], "height": 1.0}, {"t": 1.0, "at": [0.25, 0.25], "height": 0.25, "ease": "linear"}]
    hand = {"kind": "cutout", "from": 0.5, "to": 1.5, "clip": "clip.mp4", "at": {"table": [0.25, 0.25]}, "size": 0.1}
    cup_clip(tmp_path / "clip.mp4")
    with pytest.raises(TR.TransitionError, match="no table"):
        plan([table(view=view, to=1.0), hand])
    ffx, pl, _ = frames(tmp_path, [table(view=view), hand])
    assert pl["fx"][1]["at"]["tr"] == {"sheet": 0}
    p, zoom, turn = ffx.table_point(0, [0.25, 0.25], F0 + 30)
    assert p == pytest.approx(TB.middle((W, H)), abs=1e-6) and turn == pytest.approx(0.0)
    assert zoom == pytest.approx(4.0) and math.isclose(ffx.table_point(0, [0.25, 0.25], F0)[1], 1.0)
