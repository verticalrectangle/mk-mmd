"""Cut effects composited (mkmmd.cutfx) from synthetic layers on disk: flat colours, a cross as the figure, tiny frames. Checks
which shot shows inside the matte and which outside, the rim, that an expand fills the frame on its last frame (and a collapse
on its first) whatever scale the spec asks, the slash band, the thought bubble through their pop, hold and exit, and a glitch
breaking the figure up until its bare frame is left."""
import json

import cv2
import numpy as np
import pytest

from mkmmd import cutfx as CF
from mkmmd import post as P
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

W, H, FPS, F0 = 96, 54, 30, 1
MOON = PAL.get("rose-pine-moon")


def q8(c):
    return tuple(round(v * 255) / 255 for v in c)                  # what a PNG holds: layers round-trip exactly


COL = {k: q8(v) for k, v in {"s1": (0.90, 0.30, 0.40), "l1": (0.20, 0.70, 0.30), "s2": (0.95, 0.75, 0.40),
                             "l2": (0.20, 0.40, 0.90), "s3": (0.60, 0.80, 0.85), "pl": (0.60, 0.30, 0.70)}.items()}
FIG = q8((0.10, 0.10, 0.20))
TEXT = PAL.srgb(MOON["text"])
SIL = {"style": "silhouette"}


def cross(ss=4, big=2):
    """Coverage of a cross (arms 10 px wide, 30 long) at the frame's middle, drawn `big` times the frame's size."""
    w, h = W * big, H * big
    img = np.zeros((h * ss, w * ss), np.uint8)
    cx, cy, k = W * big * ss // 2, H * big * ss // 2, big * ss
    cv2.rectangle(img, (cx - 5 * k, cy - 15 * k), (cx + 5 * k - 1, cy + 15 * k - 1), 255, -1)
    cv2.rectangle(img, (cx - 15 * k, cy - 5 * k), (cx + 15 * k - 1, cy + 5 * k - 1), 255, -1)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def flat(colour):
    return np.broadcast_to(np.asarray(colour, np.float32), (H, W, 3)).copy()


def silhouette(name):
    a = cross(big=1)[..., None]
    return flat(COL[name]) * (1 - a) + np.asarray(FIG, np.float32) * a


def picture(name):
    return silhouette(name) if name.startswith("s") else flat(COL[name])


def save_rgb(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))


def save_gray(path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def project(**extra):
    data = {"output": [{"name": "t"}],
            "shot": [{"name": "s1", "from": 0.0, "to": 2.0, **SIL}, {"name": "l1", "from": 2.0, "to": 4.0},
                     {"name": "s2", "from": 4.0, "to": 6.0, **SIL}, {"name": "l2", "from": 6.0, "to": 8.0},
                     {"name": "s3", "from": 8.0, "to": 10.0, **SIL}, {"name": "pl", "plate": True}],
            "transition": [{"at": 2.0, "kind": "expand", "scale": [1, 4], "edge": {"color": "text", "width": 80}},
                           {"at": 4.0, "kind": "collapse", "edge": {"color": "text", "width": 80}},
                           {"at": 6.0, "kind": "slash", "color": "love", "width": 0.25,
                            "second": {"color": "text", "width": 0.04, "offset": -0.04}}],
            "insert": [{"from": 6.4, "to": 8.0, "shot": "s3", "anchor": "x", "size": 0.5, "offset": [0.0, -0.55],
                        "out": "expand", "outline": {"color": "text", "width": 80}},
                       {"from": 8.2, "to": 9.6, "shot": "pl", "anchor": "x", "size": 0.5, "offset": [0.0, -0.55],
                        "out": "pop", "outline": {"color": "text", "width": 80}}],
            "glitch": [{"from": 0.3, "to": 0.7, "objects": ["x*"]}]}
    data.update(extra)
    return data


def build(root, data, anchor=(0.5, 0.95), per_metre=0.5):
    """Write the cut's frames and every layer the plan demands; returns (plan, CutFx)."""
    plan = TR.plan(data, FPS, F0, MOON)
    for f in range(F0, F0 + 300):
        save_rgb(root / f"{f:05d}.png", picture(TR.cut_shot(plan["cuts"], f)))
    for f, items in TR.demands(plan).items():
        for it in items:
            if it["kind"] == "plate":
                save_rgb(root / TR.plate_rel(it["shot"], f), picture(it["shot"]))
            elif it["kind"] == "matte":
                save_gray(root / TR.matte_rel(it["shot"], f), cross())
                save_rgb(root / TR.back_rel(it["shot"], f), flat(COL[it["shot"]]))
            elif it["kind"] == "bare":
                save_rgb(root / TR.bare_rel(it["key"], f), flat(COL[it["shot"]]))     # the shot without its figure
            else:
                (root / TR.point_rel(it["key"], f)).parent.mkdir(parents=True, exist_ok=True)
                (root / TR.point_rel(it["key"], f)).write_text(json.dumps({"p": list(anchor), "depth": 5.0, "m": per_metre}))
    return plan, CF.CutFx(plan, root, (W, H), "t")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("frames")
    plan, fx = build(root, project())
    return root, plan, fx


def frame(world, f):
    root, _, fx = world
    return fx.frame(f, P.read(root / f"{f:05d}.png"))


def px(img, x, y):
    return img[y, x]


CENTRE = (W // 2, H // 2)


# ---------------------------------------------------------------- frames outside the windows
def test_a_frame_outside_every_window_is_returned_untouched(world):
    root, plan, fx = world
    main = P.read(root / f"{F0 + 5:05d}.png")
    assert not fx.active(F0 + 5) and fx.frame(F0 + 5, main) is main


def test_nothing_is_missing_and_a_deleted_layer_is_found(world, tmp_path):
    root, plan, fx = world
    assert fx.missing(range(F0, F0 + 300)) == []
    plan2, fx2 = build(tmp_path, project())
    (tmp_path / TR.plate_rel("l1", plan2["transitions"][0]["first"])).unlink()
    assert fx2.missing(range(F0, F0 + 300)) == [TR.plate_rel("l1", plan2["transitions"][0]["first"])]
    (tmp_path / TR.matte_rel("s1", plan2["transitions"][0]["first"] + 1)).write_bytes(b"")
    assert TR.matte_rel("s1", plan2["transitions"][0]["first"] + 1) in fx2.missing(range(F0, F0 + 300))


# ---------------------------------------------------------------- glitch
def test_a_glitch_breaks_the_figure_up_over_its_window_and_leaves_its_bare_frame_on_the_last(world, tmp_path):
    root, plan, fx = world
    g = plan["glitches"][0]
    assert g["host"] == "s1" and fx.active(g["f0"]) and fx.active(g["f1"] - 1) and not fx.active(g["f1"])
    assert frame(world, g["f1"] - 1) == pytest.approx(flat(COL["s1"]), abs=0.01)          # the cross gone, nothing left
    broken = [f for f in range(g["f0"], g["f1"] - 1)
              if not np.allclose(frame(world, f), silhouette("s1"), atol=0.01)
              and not np.allclose(frame(world, f), flat(COL["s1"]), atol=0.01)]
    assert len(broken) >= (g["f1"] - g["f0"]) // 2
    plan2, fx2 = build(tmp_path, project())
    (tmp_path / TR.bare_rel("g0", g["f0"] + 2)).unlink()
    assert fx2.missing(range(F0, F0 + 300)) == [TR.bare_rel("g0", g["f0"] + 2)]


# ---------------------------------------------------------------- expand
def test_the_first_frame_of_an_expand_shows_the_incoming_shot_in_the_figures_shape_and_a_rim_round_it(world):
    _, plan, _ = world
    t = plan["transitions"][0]
    img = frame(world, t["first"])
    assert px(img, *CENTRE) == pytest.approx(COL["l1"], abs=0.01)           # inside the figure: the incoming shot
    assert px(img, 5, 5) == pytest.approx(COL["s1"], abs=0.01)              # outside: the outgoing frame without the figure
    assert px(img, CENTRE[0] + 16, CENTRE[1]) == pytest.approx(TEXT, abs=0.02)      # the rim, 1-4 px off the arm's end
    assert px(img, CENTRE[0] + 25, CENTRE[1]) == pytest.approx(COL["s1"], abs=0.01)  # and none beyond it
    assert px(img, CENTRE[0] + 14, CENTRE[1] - 14) == pytest.approx(COL["s1"], abs=0.01)   # a notch of the cross: not a box


def test_an_expand_fills_the_frame_with_the_incoming_shot_on_its_last_frame(world):
    _, plan, _ = world
    t = plan["transitions"][0]
    last = frame(world, t["last"])
    assert np.abs(last - flat(COL["l1"])).max() < 1e-4


def test_an_expand_grows_every_frame_and_the_matte_turns(world):
    _, plan, _ = world
    t = plan["transitions"][0]
    areas = []
    for f in range(t["first"], t["last"] + 1):
        img = frame(world, f)
        areas.append(float((np.abs(img - np.asarray(COL["l1"])).max(axis=2) < 0.02).sum()))
    assert all(b >= 0.9 * a for a, b in zip(areas, areas[1:]))               # (edge pixels turn antialiased as it starts to turn)
    assert areas[-1] == W * H and areas[0] < areas[3] < areas[6] < areas[-1]
    mid = frame(world, t["first"] + 4)
    plain = frame(world, t["first"])
    assert np.abs(mid - plain).max() > 0.3


def test_the_top_scale_is_raised_to_what_fills_the_frame(world):
    _, plan, fx = world
    fx.check(range(F0, F0 + 300))
    note = [n for n in fx.notes if n["transition"] == 0][0]
    assert note["raised"] and note["scale"][1] == pytest.approx(note["needed"] * CF.COVER_MARGIN, rel=1e-3) and note["needed"] > 4


def test_fill_tops_the_scale_at_exactly_what_fills_the_frame(tmp_path):
    data = project()
    data["transition"][0].pop("scale")
    plan, fx = build(tmp_path, data)
    last = fx.frame(plan["transitions"][0]["last"], P.read(tmp_path / f"{plan['transitions'][0]['last']:05d}.png"))
    note = fx.notes[-1]
    assert note["transition"] == 0 and not note["raised"] and np.abs(last - flat(COL["l1"])).max() < 1e-4
    first = plan["transitions"][0]["first"]
    assert note["scale"][1] == pytest.approx(note["needed"] * CF.COVER_MARGIN, rel=1e-3)
    assert px(fx.frame(first, P.read(tmp_path / f"{first:05d}.png")), *CENTRE) == pytest.approx(COL["l1"], abs=0.01)


def test_a_centre_outside_the_figure_moves_into_it_and_still_fills_the_frame(tmp_path):
    data = project()
    data["transition"][0]["center"] = [0.02, 0.05]
    plan, fx = build(tmp_path, data)
    t = plan["transitions"][0]
    fx.check([t["last"]])
    assert np.abs(fx.frame(t["last"], P.read(tmp_path / f"{t['last']:05d}.png")) - flat(COL["l1"])).max() < 1e-4


def test_a_centre_given_as_a_point_of_the_frame_is_where_the_matte_turns(tmp_path):
    data = project()
    data["transition"][0].update(center=[0.5, 0.3], scale=[3, 3.5], turn=[0, 0])    # inside the cross's upper arm
    plan, fx = build(tmp_path, data)
    t = plan["transitions"][0]
    img = fx.frame(t["first"], P.read(tmp_path / f"{t['first']:05d}.png"))
    inside = (np.abs(img - np.asarray(COL["l1"])).max(axis=2) < 0.02)
    ys, xs = np.nonzero(inside)
    # the cross, scaled 3x about a point in its upper arm (y = 0.3 * 54 - 0.5 = 15.7): its top edge sits 3x farther from it
    assert ys.min() == pytest.approx(15.7 - 3 * (15.7 - 12.0), abs=1.5)


def test_a_figure_that_is_missing_at_the_covering_frame_is_an_error_that_says_which(tmp_path):
    plan, fx = build(tmp_path, project())
    t = plan["transitions"][0]
    save_gray(tmp_path / TR.matte_rel("s1", t["last"]), np.zeros((H * 2, W * 2), np.float32))
    with pytest.raises(TR.TransitionError, match="shows no figure at frame"):
        fx.check([t["last"]])


def test_a_projected_centre_is_read_from_the_point_layer(tmp_path):
    data = project()
    data["transition"][0].update(center='bone("x").head', scale=[3, 3.5], turn=[0, 0])
    plan, fx = build(tmp_path, data, anchor=(0.5, 0.35))
    t = plan["transitions"][0]
    assert TR.demands(plan)[t["first"]][-1] == {"kind": "point", "key": "t0", "expr": 'bone("x").head', "shot": "s1"}
    img = fx.frame(t["first"], P.read(tmp_path / f"{t['first']:05d}.png"))
    ys, xs = np.nonzero(np.abs(img - np.asarray(COL["l1"])).max(axis=2) < 0.02)
    assert ys.min() == pytest.approx(0.35 * H - 0.5 - 3 * (0.35 * H - 0.5 - 12.0), abs=1.5)


# ---------------------------------------------------------------- collapse
def test_a_collapse_starts_as_the_outgoing_frame_untouched_and_lands_on_the_figure(world):
    _, plan, _ = world
    t = plan["transitions"][1]
    first = frame(world, t["first"])
    assert np.abs(first - flat(COL["l1"])).max() < 1e-4                     # the matte fills the frame: the cut's own frame
    last = frame(world, t["last"])
    assert px(last, *CENTRE) == pytest.approx(COL["l1"], abs=0.01)          # still the outgoing shot inside the figure
    assert px(last, 5, 5) == pytest.approx(COL["s2"], abs=0.01)             # the incoming shot's background outside
    assert px(last, CENTRE[0] + 16, CENTRE[1]) == pytest.approx(TEXT, abs=0.02)


def test_a_collapse_shrinks_every_frame(world):
    _, plan, _ = world
    t = plan["transitions"][1]
    areas = [float((np.abs(frame(world, f) - np.asarray(COL["l1"])).max(axis=2) < 0.02).sum())
             for f in range(t["first"], t["last"] + 1)]
    assert all(b <= a for a, b in zip(areas, areas[1:])) and areas[0] == W * H and areas[-1] < 0.3 * W * H


# ---------------------------------------------------------------- slash
def test_a_slash_has_the_incoming_shot_behind_the_band_and_the_outgoing_one_ahead(world):
    _, plan, _ = world
    t = plan["transitions"][2]
    mid = frame(world, t["first"] + 2)                                      # the band is across the middle
    love = np.asarray(PAL.srgb(MOON["love"]))
    assert px(mid, *CENTRE) == pytest.approx(love, abs=0.02)
    assert px(mid, 2, H // 2) == pytest.approx(COL["l2"], abs=0.02)          # left: behind it
    assert px(mid, W - 3, H // 2) == pytest.approx(COL["s2"], abs=0.02) or px(mid, W - 3, H // 2) == pytest.approx(FIG, abs=0.02)
    assert (np.abs(mid - np.asarray(TEXT)).max(axis=2) < 0.02).sum() > 20    # the thin band, in its own colour


def test_a_slash_is_on_screen_in_every_window_frame_and_off_it_at_the_cut(world):
    root, plan, fx = world
    t = plan["transitions"][2]
    love = np.asarray(PAL.srgb(MOON["love"]))
    for f in range(t["first"], t["last"] + 1):
        img = frame(world, f)
        assert (np.abs(img - love).max(axis=2) < 0.02).sum() > 0               # a sliver at the first frame, more later
    assert not fx.active(t["cut"])
    done = frame(world, t["last"])
    behind = (np.abs(done - np.asarray(COL["l2"])).max(axis=2) < 0.02).mean()
    assert behind > 0.6


def test_a_slash_can_sweep_the_other_way():
    plan = TR.plan(project(transition=[{"at": 6.0, "kind": "slash", "dir": "left", "angle": 0, "width": 0.2}], insert=[]),
                   FPS, F0, MOON)
    t = plan["transitions"][0]
    assert t["dir"] == "left"


def test_the_slash_sweep_direction_follows_dir(tmp_path):
    for direction, side in (("right", 2), ("left", W - 3)):
        d = project(transition=[{"at": 6.0, "kind": "slash", "dir": direction, "angle": 0, "width": 0.2}], insert=[])
        plan, fx = build(tmp_path / direction, d)
        t = plan["transitions"][0]
        img = fx.frame(t["first"] + 3, P.read(tmp_path / direction / f"{t['first'] + 3:05d}.png"))    # late in the sweep
        assert px(img, side, 5) == pytest.approx(COL["l2"], abs=0.02)        # the side the band started from shows the new shot


# ---------------------------------------------------------------- insert
def test_the_bubble_is_a_picture_in_picture_with_an_outline_that_pops_in_holds_and_grows(world):
    root, plan, fx = world
    ins = plan["inserts"][0]
    host = COL["l2"]
    hold = frame(world, ins["f0"] + 20)
    # the bubble sits above the anchor (0.5, 0.95): centre (47.5, 0.95 * 54 - 0.5 - 0.55 * 54) = (47.5, 21.1), half-height 13.5
    bx, by = 47, 21
    assert px(hold, bx, by) == pytest.approx(FIG, abs=0.05)                  # the plate's own middle (the cross) shows in it
    assert px(hold, bx - 10, by - 8) == pytest.approx(COL["s3"], abs=0.02)   # and its background elsewhere
    assert px(hold, 2, 2) == pytest.approx(host, abs=0.01)                   # the host outside
    ring = (np.abs(hold - np.asarray(TEXT)).max(axis=2) < 0.02).sum()
    assert ring > 40                                                         # the outline (and the circles) in its colour


def test_the_first_frame_of_an_insert_has_only_the_circles_started(world):
    _, plan, _ = world
    ins = plan["inserts"][0]
    first = frame(world, ins["f0"])
    host = flat(COL["l2"])
    changed = np.abs(first - host).max(axis=2) > 0.02
    assert 0 < changed.sum() < 80                                            # a few pixels of the nearest circle
    assert px(first, 47, 21) == pytest.approx(COL["l2"], abs=0.01)           # the bubble has not begun


def test_an_expanding_insert_ends_on_the_plate_shot_itself(world):
    _, plan, _ = world
    ins = plan["inserts"][0]
    last = frame(world, ins["f1"] - 1)
    assert np.abs(last - picture("s3")).max() < 1e-3
    grow = frame(world, ins["f1"] - ins["n_exit"])                           # the first exit frame is the hold
    hold = frame(world, ins["f1"] - ins["n_exit"] - 1)
    assert np.abs(grow - hold).max() < 1e-4


def test_the_picture_grows_with_the_bubble_into_the_full_frame(world):
    _, plan, _ = world
    ins = plan["inserts"][0]
    sizes = []
    for f in range(ins["f1"] - ins["n_exit"], ins["f1"]):
        img = frame(world, f)
        sizes.append(float((np.abs(img - np.asarray(COL["s3"])).max(axis=2) < 0.02).sum()))
    assert all(b >= a for a, b in zip(sizes, sizes[1:])) and sizes[0] < 0.2 * W * H


def test_a_popping_insert_is_gone_on_its_last_frame_and_overshoots_on_its_way(world):
    root, plan, fx = world
    ins = plan["inserts"][1]
    last = frame(world, ins["f1"] - 1)
    assert np.abs(last - picture("s3")).max() < 1e-4                         # nothing of it left: the host frame
    hold = frame(world, ins["f0"] + 12)
    assert px(hold, 47, 21) == pytest.approx(COL["pl"], abs=0.01)            # the plate (flat here) fills the bubble
    sizes = [float((np.abs(frame(world, f) - np.asarray(COL["pl"])).max(axis=2) < 0.02).sum())
             for f in range(ins["f0"], ins["f0"] + ins["n_in"])]
    assert max(sizes) > 1.05 * sizes[-1] > 1.05 * sizes[0]                  # it swells past its size before settling


def test_the_bubble_stays_inside_the_frame_when_the_anchor_is_at_its_edge(tmp_path):
    plan, fx = build(tmp_path, project(), anchor=(0.02, 0.05))
    ins = plan["inserts"][0]
    img = fx.frame(ins["f0"] + 20, P.read(tmp_path / f"{ins['f0'] + 20:05d}.png"))
    cols = np.nonzero((np.abs(img - np.asarray(COL["l2"])).max(axis=2) > 0.05).any(axis=0))[0]
    assert cols.min() >= 0 and cols.max() < W and (img != flat(COL["l2"])).any()


def test_per_aspect_geometry_changes_the_bubble(tmp_path):
    d = project()
    d["insert"][0]["aspect"] = {"t": {"size": 0.2}}
    plan, fx = build(tmp_path, d)
    ins = plan["inserts"][0]
    img = fx.frame(ins["f0"] + 20, P.read(tmp_path / f"{ins['f0'] + 20:05d}.png"))
    big = frame_of_default = None
    plan2, fx2 = build(tmp_path / "default", project())
    img2 = fx2.frame(ins["f0"] + 20, P.read(tmp_path / "default" / f"{ins['f0'] + 20:05d}.png"))
    assert (np.abs(img - flat(COL["l2"])).max(axis=2) > 0.05).sum() < (np.abs(img2 - flat(COL["l2"])).max(axis=2) > 0.05).sum()
    assert big is None and frame_of_default is None


# ---------------------------------------------------------------- the trail of the thought bubble
def test_the_trail_runs_from_the_edge_of_the_head_to_the_cloud_with_growing_circles(tmp_path):
    fx = CF.CutFx(TR.plan(project(), FPS, F0, MOON), tmp_path, (960, 540), "t")
    a, b = 120.0, 80.0
    start, centre = (100.0, 300.0), (100.0, 100.0)                         # the cloud straight above where the trail begins
    dots = fx._dots(start, centre, a, b)
    assert len(dots) == 3
    ys = [y for _, y, _ in dots]
    rs = [r for _, _, r in dots]
    assert all(x == pytest.approx(100.0) for x, _, _ in dots)              # on the line between them
    assert ys[0] > ys[1] > ys[2] > centre[1] + b                           # nearest the head first, none inside the cloud
    assert rs[0] < rs[1] < rs[2]                                           # and growing toward the cloud
    assert start[1] - (ys[0] + rs[0]) > 0                                  # the first circle begins beyond the head's edge ...
    assert start[1] - ys[0] < 0.2 * (start[1] - centre[1] - b)             # ... right at it
    assert (ys[2] - rs[2]) - (centre[1] + b) > 0                           # the last does not touch the cloud


def test_the_trail_has_no_circles_when_the_head_reaches_the_cloud(tmp_path):
    fx = CF.CutFx(TR.plan(project(), FPS, F0, MOON), tmp_path, (960, 540), "t")
    assert fx._dots((100.0, 170.0), (100.0, 100.0), 120.0, 80.0) == []     # 10 px between the head's edge and the rim: too few


def test_no_circle_lies_on_the_head(tmp_path):
    data = project()
    for ins in data["insert"]:
        ins["radius"] = 0.11
        ins["size"] = 0.3
    plan, fx = build(tmp_path, data, per_metre=1.5)                         # the head: 0.11 m * 1.5 * 54 px = 8.9 px about the anchor
    ins = plan["inserts"][0]
    img = fx.frame(ins["f0"] + 20, P.read(tmp_path / f"{ins['f0'] + 20:05d}.png"))
    anchor = (0.5 * W - 0.5, 0.95 * H - 0.5)
    ys, xs = np.mgrid[0:H, 0:W]
    on_head = np.hypot(xs - anchor[0], ys - anchor[1]) < 8.9 - 1.0
    changed = np.abs(img - flat(COL["l2"])).max(axis=2) > 0.02
    assert not (changed & on_head).any()                                   # the host's head is left alone
    assert (changed & (np.hypot(xs - anchor[0], ys - anchor[1]) < 24)).any()   # but the trail is there, just beyond it


def test_a_radius_of_zero_starts_the_trail_at_the_anchor(tmp_path):
    data = project()
    data["insert"][0].update(radius=0.0, size=0.3)
    plan, fx = build(tmp_path, data, per_metre=1.5)
    ins = plan["inserts"][0]
    img = fx.frame(ins["f0"] + 20, P.read(tmp_path / f"{ins['f0'] + 20:05d}.png"))
    anchor = (0.5 * W - 0.5, 0.95 * H - 0.5)
    ys, xs = np.mgrid[0:H, 0:W]
    changed = np.abs(img - flat(COL["l2"])).max(axis=2) > 0.02
    assert (changed & (np.hypot(xs - anchor[0], ys - anchor[1]) < 7)).any()


def test_a_point_layer_from_before_its_scale_was_stored_is_missing_and_unusable(tmp_path):
    plan, fx = build(tmp_path, project())
    ins = plan["inserts"][0]
    rel = TR.point_rel("i0", ins["f0"] + 3)
    (tmp_path / rel).write_text(json.dumps({"p": [0.5, 0.9], "depth": 5.0}))      # the old format
    assert rel in fx.missing(range(F0, F0 + 300))
    with pytest.raises(FileNotFoundError, match="before its scale"):
        fx.frame(ins["f0"] + 3, P.read(tmp_path / f"{ins['f0'] + 3:05d}.png"))
    TR.clear_claims(tmp_path)                                               # what mk render does first: the stale file goes
    assert not (tmp_path / rel).exists() and (tmp_path / TR.point_rel("i0", ins["f0"] + 4)).exists()
