"""Googly eyes on sprites (mkmmd.core.eyes, mkmmd.eyes and the shared sprite path of mkmmd.mandala): the face keys, where
the eyes sit on a sprite's alpha or its registry anchors, the pupils' swing (deterministic, kicked by hits and motion,
kept inside the eye), and the eyes drawn on tiles and on a mandala's rings."""
import numpy as np
import pytest
from PIL import Image

from mkmmd import assets as A
from mkmmd import eyes as EYD
from mkmmd import framefx as FF
from mkmmd import mandala as MDF
from mkmmd.core import eyes as EY
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")


def face(**eyes):
    return EY.face({"eyes": eyes or True}, "test")


def disc_alpha(h=200, w=200, r=80, cx=None, cy=None):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx = (w - 1) / 2.0 if cx is None else cx
    cy = (h - 1) / 2.0 if cy is None else cy
    return np.clip(r + 0.5 - np.hypot(xx - cx, yy - cy), 0.0, 1.0)


def covered(alpha, x, y):
    h, w = alpha.shape
    return alpha[int(round(y * h - 0.5)), int(round(x * w - 0.5))] > 0.5


def sprite_png(path, alpha, colour=(200, 40, 40)):
    a = (np.asarray(alpha) * 255).astype(np.uint8)
    rgb = np.dstack([np.full(a.shape, c, np.uint8) for c in colour])
    Image.fromarray(np.dstack([rgb, a]), "RGBA").save(path)
    return str(path)


# ---------------------------------------------------------------- the keys
def test_eyes_true_fills_in_the_defaults_and_a_mouth_alone_is_a_face():
    f = EY.face({"eyes": True}, "t")
    assert f["eyes"] and not f["mouth"] and f["size"] == 1.0 and f["look"] == "down" and f["seed"] == 1
    assert f["amount"] == EY.WOBBLE["amount"] and f["times"] is None and f["lane"] is None
    assert EY.face({}, "t") is None and EY.face({"eyes": False}, "t") is None
    m = EY.face({"mouth": True}, "t")
    assert m["mouth"] and not m["eyes"]
    w = EY.face({"eyes": {"wobble": {"amount": 2.0, "hz": 4.0}, "look": [0.2, 0.8], "lane": "kick", "seed": 3}}, "t")
    assert (w["amount"], w["hz"], w["look"], w["lane"], w["seed"]) == (2.0, 4.0, [0.2, 0.8], "kick", 3)
    assert EY.face({"eyes": {"wobble": 0.5}}, "t")["amount"] == 0.5


@pytest.mark.parametrize("src", [{"eyes": "yes"}, {"eyes": {"blink": 1}}, {"eyes": {"look": "up"}},
                                 {"eyes": {"size": 0}}, {"mouth": "smile"}, {"eyes": {"wobble": {"spring": 1}}},
                                 {"eyes": {"times": [1.0], "lane": "kick"}}])
def test_bad_faces_are_errors(src):
    with pytest.raises(FX.FxError):
        EY.face(src, "t")


def test_a_ring_of_sprites_takes_eyes_and_a_mouth_and_nothing_else_new():
    spec = {"kind": "mandala", "from": 0.0, "to": 1.0,
            "rings": [{"source": {"sprites": "a.png", "eyes": {"size": 1.5}, "mouth": True}}]}
    pl = TR.plan({"project": {"duration": 2.0}, "output": [{"name": "o", "size": [64, 36]}],
                  "shot": [{"name": "a", "from": 0.0, "to": 2.0}], "fx": [spec]}, FPS, F0, DAWN)
    src = pl["fx"][0]["rings"][0]["source"]
    assert src["face"]["size"] == 1.5 and src["face"]["mouth"]
    spec["rings"][0]["source"]["nose"] = True
    with pytest.raises(TR.TransitionError):
        TR.plan({"project": {"duration": 2.0}, "output": [{"name": "o", "size": [64, 36]}],
                 "shot": [{"name": "a", "from": 0.0, "to": 2.0}], "fx": [spec]}, FPS, F0, DAWN)


# ---------------------------------------------------------------- where they sit
def test_a_round_sprite_gets_its_eyes_side_by_side_above_its_middle_on_it_and_a_smile_under_them():
    a = disc_alpha()
    got = EY.auto_face(a)
    (x1, y1), (x2, y2) = got["at"]
    assert y1 == pytest.approx(y2) and x1 < 0.5 < x2
    assert (x1 + x2) / 2 == pytest.approx(0.5, abs=0.02)            # centred across it
    assert 0.25 < y1 < 0.5                                          # the upper middle of the widest part
    assert EY.EYE_R[1] - 1e-9 <= got["r"] <= EY.EYE_R[2] + 1e-9
    assert covered(a, x1, y1) and covered(a, x2, y2)
    assert got["mouth"][1] > y1 and covered(a, *got["mouth"])


def test_the_eyes_go_to_the_widest_part_of_a_carrot_or_a_broccoli():
    h, w = 400, 200
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    carrot = (np.abs(xx - 99.5) < 60.0 * (1.0 - yy / h)).astype(float)           # wide at the top, a point below
    got = EY.auto_face(carrot)
    assert got["at"][0][1] < 0.35 and all(covered(carrot, *p) for p in got["at"])
    crown = np.hypot(xx - 99.5, (yy - 100) * 1.6) < 95                           # a wide crown on a narrow stem
    stem = (np.abs(xx - 99.5) < 22) & (yy > 100)
    broccoli = (crown | stem).astype(float)
    got = EY.auto_face(broccoli)
    assert got["at"][0][1] < 0.35 and all(covered(broccoli, *p) for p in got["at"])
    assert got["at"][1][0] - got["at"][0][0] > 0.2                               # spread over the crown, not the stem


def test_registry_anchors_win_over_the_auto_face_and_fill_in_what_they_leave_out():
    a = disc_alpha()
    full = EY.anchors({"at": [[0.3, 0.6], [0.7, 0.6]], "r": 0.05, "mouth": False}, a)
    assert full == {"at": [[0.3, 0.6], [0.7, 0.6]], "r": 0.05, "mouth": None}
    part = EY.anchors({"r": 0.2}, a)
    assert part["r"] == 0.2 and part["at"] == EY.auto_face(a)["at"]
    for bad in ({"at": [[0.3, 0.6]]}, {"r": 2.0}, {"eyes": 1}, {"at": [[5, 5], [6, 6]]}, "left"):
        with pytest.raises(FX.FxError):
            EY.anchors(bad, a)


# ---------------------------------------------------------------- how they swing
def still(face, t, n_eyes=2, at=(100.0, 100.0), free=10.0):
    ts = EY.window(face, t)
    c = np.broadcast_to(np.asarray(at, float), (len(ts), n_eyes, 2))
    return ts, c, np.full((len(ts), n_eyes), free)


def test_the_window_ends_on_the_frame_on_a_grid_aligned_to_clip_second_0():
    f = face()
    ts = EY.window(f, 2.0)
    assert ts[-1] == 2.0 and np.allclose(np.diff(ts), 1.0 / EY.RATE)
    decay = 1.0 / (2 * np.pi * f["hz"] * f["damping"])
    assert ts[-1] - ts[0] == pytest.approx(min(EY.SETTLE * decay, EY.MAX_WINDOW), abs=1.0 / EY.RATE)
    odd = EY.window(f, 2.0 + 1.0 / 48.0)                            # a frame between the grid's samples comes last
    assert odd[-1] == pytest.approx(2.0 + 1.0 / 48.0) and odd[-2] == pytest.approx(np.floor((2.0 + 1 / 48.0) * EY.RATE) / EY.RATE)


def test_still_pupils_without_hits_rest_where_gravity_drops_them_or_look_out():
    f = face()
    ts, c, free = still(f, 3.0)
    assert np.allclose(EY.swing(f, ts, c, free), [[0.0, EY.SAG]] * 2)
    assert np.allclose(EY.swing(f, ts, c, free, look="viewer"), 0.0)
    toward = EY.swing(f, ts, c, free, look=(200.0, 100.0))           # a point to the right
    assert np.allclose(toward, [[EY.GAZE, 0.0]] * 2)


def test_a_hit_kicks_the_pupils_up_they_swing_inside_the_eye_and_settle_back():
    f = face()
    hit = 2.0
    kicked = []
    for t in np.arange(2.0, 4.6, 1.0 / FPS):
        ts, c, free = still(f, t)
        p = EY.swing(f, ts, c, free, hits=[hit], keys=[[0, 0], [0, 1]])
        assert np.all(np.linalg.norm(p, axis=1) <= 1.0 + 1e-9)       # never out of the eye
        kicked.append(p)
    kicked = np.array(kicked)
    assert np.allclose(kicked[0], [[0.0, EY.SAG]] * 2)               # on the hit's frame: not moved yet
    assert kicked[1:6, :, 1].min() < EY.SAG - 0.5                    # thrown up
    assert np.abs(np.diff(kicked[:30, 0, 1])).max() > 0.05           # it jiggles from frame to frame
    assert np.allclose(kicked[-1], [[0.0, EY.SAG]] * 2, atol=0.02)   # settled 2.5 s later
    assert not np.allclose(kicked[3, 0], kicked[3, 1])                # the two eyes not in lockstep


def test_the_swing_is_a_function_of_the_clip_second_and_the_seed_alone():
    f = face()
    hits = [1.0, 1.4, 2.2]
    times = [2.5, 1.1, 2.3, 1.5, 2.5]
    def at(t, seed=1, keys=((3, 1, 0), (3, 1, 1))):
        ts, c, free = still(dict(f, seed=seed), t)
        return EY.swing(dict(f, seed=seed), ts, c, free, hits=hits, keys=keys)
    first = {t: at(t) for t in times}
    again = {t: at(t) for t in reversed(times)}                     # any order, any number of times
    assert all(np.array_equal(first[t], again[t]) for t in times)
    assert not np.allclose(at(2.5), at(2.5, seed=2))                 # another seed, another take
    assert not np.allclose(at(2.5), at(2.5, keys=((3, 2, 0), (3, 2, 1))))   # another copy, its own kicks
    assert np.array_equal(EY.kicks(f, 1.4, [[3, 1, 0]]), EY.kicks(f, 1.4, [[3, 1, 0]]))


def test_motion_shakes_the_pupils_the_other_way_when_the_eye_starts_or_stops():
    f = dict(face(), damping=0.3)
    stop = 2.0

    def moving(t):                                                  # eyes sliding right at 300 px/s until 2 s
        ts = EY.window(f, t)
        x = 100.0 + 300.0 * np.minimum(ts, stop)
        c = np.stack([np.stack([x, np.full_like(x, 50.0)], -1)] * 2, axis=1)
        return EY.swing(f, ts, c, np.full((len(ts), 2), 8.0))
    assert np.allclose(moving(1.5), [[0.0, EY.SAG]] * 2, atol=1e-6)          # steady motion: no force, at rest
    after = np.array([moving(stop + k / FPS) for k in range(1, 8)])
    assert after[:, 0, 0].max() > 0.3                                         # it stopped: the pupils fly on (right)
    assert np.allclose(moving(stop + 3.0), [[0.0, EY.SAG]] * 2, atol=0.02)


def test_no_wobble_holds_the_pupils_at_rest():
    f = face(wobble=0)
    ts, c, free = still(f, 2.5)
    assert np.allclose(EY.swing(f, ts, c, free, hits=[2.0, 2.4]), [[0.0, EY.SAG]] * 2)


# ---------------------------------------------------------------- drawn
def test_a_sprite_tile_with_eyes_has_white_eyes_where_its_anchors_say_and_grows_to_hold_them(tmp_path):
    a = disc_alpha(120, 120, 50)
    path = sprite_png(tmp_path / "s.png", a)
    MDF._resolve([path], None, MDF._registry(None), True)
    bare = MDF.sprite_tile(path, 120, 0.0)
    tl = MDF.sprite_tile(path, 120, 0.0, face=EY.face({"eyes": True, "mouth": True}, "t"))
    assert bare.face is None and tl.face is not None
    for x, y in tl.face["eyes"]:                                    # a white dome inside a dark rim
        assert tl.a[int(round(y)), int(round(x))] == pytest.approx(1.0)
        assert tl.rgb[int(round(y - 0.4 * tl.face["inner"])), int(round(x - 0.3 * tl.face["inner"]))].min() > 0.95
        assert tl.rgb[int(round(y)), int(round(x + tl.face["R"] - 0.5))].max() < 0.35
    mx, my = tl.face["mouth"]                                       # the smile's middle is drawn on its anchor
    assert tl.rgb[int(round(my)), int(round(mx))].max() < 0.5
    big = MDF.sprite_tile(path, 120, 0.0, face=EY.face({"eyes": {"size": 3.0}}, "t"))
    assert big.a.shape[1] > bare.a.shape[1]                          # eyes sticking out widen the tile


def test_lay_draws_the_pupils_at_rest_below_the_eye_middles_and_where_offsets_put_them(tmp_path):
    path = sprite_png(tmp_path / "s.png", disc_alpha(120, 120, 50))
    tl = MDF.sprite_tile(path, 120, 0.0, face=EY.face({"eyes": {"size": 1.5}}, "t"))
    M = MDF.placed(tl, (100.0, 100.0), 0.0, 1.0)
    (ex, ey), free = (M[:, :2] @ np.asarray(tl.face["eyes"][0]) + M[:, 2]), tl.face["free"]

    def dark(off):
        out = MDF.lay(np.full((200, 200, 3), 0.5, np.float32), tl, M, pupils=off)
        return out[int(round(ey + off[0][1] * free)), int(round(ex + off[0][0] * free))].max() < 0.15
    assert dark(np.array([[0.0, EY.SAG], [0.0, EY.SAG]]))
    assert dark(np.array([[0.9, 0.0], [0.9, 0.0]])) and dark(np.array([[0.0, -0.9], [0.0, -0.9]]))
    out = MDF.lay(np.full((200, 200, 3), 0.5, np.float32), tl, M)   # no offsets: at rest
    assert out[int(round(ey + EY.SAG * free)), int(round(ex))].max() < 0.15
    moved = EYD.with_pupils(tl, np.array([[-0.9, 0.0], [-0.9, 0.0]]))
    x0, y0 = tl.face["eyes"][0]
    assert moved.rgb[int(round(y0)), int(round(x0 - 0.9 * free))].max() < 0.15
    assert np.array_equal(moved.a, tl.a)                             # pupils inside the whites: coverage unchanged


def test_registry_extra_eyes_place_a_registered_sprite_s_eyes(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    path = sprite_png(tmp_path / "veg.png", disc_alpha(100, 100, 45))
    reg = A.Registry()
    reg.add({"kind": "texture", "slug": "veg", "path": path, "tags": ["sprite", "veggie"],
             "extra": {"eyes": {"at": [[0.3, 0.7], [0.62, 0.7]], "r": 0.06, "mouth": False}}})
    reg.save()
    files = MDF.sprite_source(tmp_path)([{"tag": ["veggie"]}])
    tl = MDF.sprite_tile(files[0], 100, 0.0, face=EY.face({"eyes": True, "mouth": True}, "t"))
    pad = 2
    assert tl.face["eyes"][0] == pytest.approx([0.3 * 100 - 0.5 + pad, 0.7 * 100 - 0.5 + pad])
    assert tl.face["R"] == pytest.approx(6.0) and tl.face["mouth"] is None    # this sprite takes no smile


def test_mk_assets_set_merges_a_json_extra_over_the_entry_s_and_a_null_removes_a_key():
    from mkmmd.cli.assets import _extra
    from mkmmd.cli.common import UsageError
    eyes = {"at": [[0.4, 0.3], [0.6, 0.3]], "r": 0.05}
    assert _extra({"pexels_id": 7}, '{"eyes": %s}' % str(eyes).replace("'", '"')) == {"pexels_id": 7, "eyes": eyes}
    assert _extra({"pexels_id": 7, "eyes": eyes}, '{"eyes": null}') == {"pexels_id": 7}
    assert _extra({"eyes": eyes}, '{"eyes": null}') is None
    for bad in ("{eyes", "[1, 2]"):
        with pytest.raises(UsageError):
            _extra(None, bad)


def test_a_mandala_of_sprites_with_eyes_jiggles_its_pupils_on_the_hits_the_same_every_time(tmp_path):
    path = sprite_png(tmp_path / "s.png", disc_alpha(120, 120, 55), colour=(60, 160, 60))
    spec = {"kind": "mandala", "from": 1.0, "to": 3.0, "back": "#000000", "times": [1.5],
            "rings": [{"count": 3, "radius": 0.25, "scale": 0.3, "turn": 0.0,
                       "source": {"sprites": path, "eyes": {"size": 1.3}}}]}
    size = (200, 200)
    pl = TR.plan({"project": {"duration": 4.0}, "output": [{"name": "o", "size": list(size)}],
                  "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": [spec]}, FPS, F0, DAWN)

    def frames(order):
        run = FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                         sprite_of=MDF.sprite_source())
        return {g: run.frame(g, np.zeros((200, 200, 3), np.float32)) for g in order}
    gs = [F0 + 40, F0 + 44, F0 + 46, F0 + 47, F0 + 48, F0 + 53, F0 + 49]          # the hit at 1.5 s is frame F0 + 45
    a, b = frames(gs), frames(list(reversed(gs)))
    assert all(np.array_equal(a[g], b[g]) for g in gs)               # deterministic, in any order
    assert np.array_equal(a[F0 + 46], frames([F0 + 46])[F0 + 46])    # and alone
    assert np.array_equal(a[F0 + 40], a[F0 + 44])                    # still before the hit
    assert all(np.abs(a[g] - a[F0 + 40]).max() > 0.5 for g in (F0 + 46, F0 + 47, F0 + 48))   # thrown by the hit
    assert np.abs(a[F0 + 47] - a[F0 + 49]).max() > 0.5 and np.abs(a[F0 + 49] - a[F0 + 53]).max() > 0.5   # jiggling
