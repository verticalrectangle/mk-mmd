"""Ring portals (`[[insert]] shape = "iris"`, `ring = "vinyl" | "plate" | "lens"`; mkmmd.core.portal, mkmmd.portal and
their hooks in mkmmd.core.iris, mkmmd.iris and mkmmd.cutfx): a ring's spec, defaults and errors, the rim defaults it
sets, its layout in window radii, the record's spin and the plate's seats, and the composite on tiny synthetic frames
through CutFx: nothing past the ring's reach, the window still the iris's, an expand still landing on the inner picture,
a dark record round a label turning clockwise at 33 1/3 under a sheen that stays in the light, a plate lit from the
upper left with sprites (and googly eyes) round its rim, lens rings that show the inner picture with gaps between them.
Fixtures use made-up shot names and a sprite drawn here."""
import json
import math
import re

import cv2
import numpy as np
import pytest

from mkmmd import cutfx as CF
from mkmmd import matte as M
from mkmmd import portal as PORTAL
from mkmmd.core import framefx as FX
from mkmmd.core import iris as IR
from mkmmd.core import palette as PAL
from mkmmd.core import portal as PT
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
ANCHOR = 'obj("dish").loc'
IRIS = {"from": 1.0, "to": 3.0, "shot": "p", "shape": "iris", "anchor": ANCHOR}
EXPAND = {"from": 3.0, "to": 4.0, "shot": "n", "shape": "iris", "anchor": ANCHOR, "out": "expand"}
SIZES = {"16x9": (192, 108), "1x1": (108, 108), "9x16": (108, 192)}
RINGS = ("vinyl", "plate", "lens")


def project(inserts):
    """h 0-4 s (the host), n 4-6 s, and a plate shot p; the three outputs."""
    return {"output": [{"name": o} for o in SIZES],
            "shot": [{"name": "h", "from": 0.0, "to": 4.0}, {"name": "n", "from": 4.0, "to": 6.0}, {"name": "p", "plate": True}],
            "insert": list(inserts)}


def ins(ring, spec=IRIS, **kw):
    return TR.normalize_insert({**spec, "ring": ring, **kw}, DAWN)


def srgb(slot):
    return pytest.approx(PAL.srgb(DAWN[slot]), abs=1e-6)


# ---------------------------------------------------------------- the spec
def test_a_ring_is_a_kind_or_a_table_and_gets_its_kinds_defaults():
    v = ins("vinyl")["ring"]
    assert (v["kind"], v["size"], v["rpm"], v["sheen"]) == ("vinyl", 2.8, pytest.approx(100.0 / 3.0), 0.8)
    assert v["label"] == [srgb("gold"), srgb("love"), srgb("text")]
    assert 0.2126 * v["color"][0] + 0.7152 * v["color"][1] + 0.0722 * v["color"][2] < 0.12     # near black
    p = ins({"kind": "plate"})["ring"]
    assert (p["kind"], p["size"], p["sprites"], p["face"]) == ("plate", 1.46, None, None)
    assert p["color"] == srgb("surface") and p["band"] == srgb("pine")                       # Dawn's lightest, its pine
    assert ins({"kind": "plate", "band": False})["ring"]["band"] is None
    lens = ins("lens")["ring"]
    assert (lens["size"], lens["count"], lens["chroma"], lens["turn"]) == (2.1, 4, 5.0, 30.0)
    assert lens["tint"] == srgb("iris") and len(lens["bands"]) == 4
    assert ins(None)["ring"] is None and TR.normalize_insert(IRIS, DAWN)["ring"] is None


def test_a_plate_takes_sprites_in_a_mandalas_forms_with_eyes():
    p = ins({"kind": "plate", "sprites": [{"tag": "veggie"}, "carrot_01"], "eyes": True, "count": 6, "turn": -20})["ring"]
    assert p["sprites"] == [{"tag": ["veggie"]}, "carrot_01"] and (p["count"], p["turn"]) == (6, -20.0)
    assert p["face"]["eyes"] is True and p["face"]["mouth"] is False


def test_a_ring_sets_the_rims_defaults_under_the_inserts_own():
    plain = TR.normalize_insert(IRIS, DAWN)["rim"]
    assert (plain["width"], plain["dark"], plain["glint"], plain["fringe"]) == (0.3, 0.8, 0.85, 2.0)
    for kind in RINGS:
        rim = ins(kind)["rim"]
        assert {k: rim[k] for k in PT.RIMS[kind]} == PT.RIMS[kind]
    rim = ins("vinyl", rim={"dark": 0.9})["rim"]
    assert (rim["dark"], rim["width"], rim["glint"]) == (0.9, 0.12, 0.0)                      # its key over the ring's
    assert ins("plate", rim=False)["rim"] is None


@pytest.mark.parametrize("ring, frag", [
    ("cd", "ring.kind = 'cd'"),
    (3, "ring = 3"),
    ({"size": 2}, "ring.kind = None"),
    ({"kind": "vinyl", "count": 3}, "unknown keys ['count']"),
    ({"kind": "vinyl", "size": 1.0}, "size = 1.0"),
    ({"kind": "vinyl", "rpm": "fast"}, "rpm = 'fast'"),
    ({"kind": "vinyl", "label": ["gold", "love", "text", "pine"]}, "label = "),
    ({"kind": "vinyl", "label": ["nope"]}, "label[0]"),
    ({"kind": "vinyl", "sheen": 2}, "sheen = 2.0"),
    ({"kind": "plate", "count": 5}, "give `sprites` too"),
    ({"kind": "plate", "eyes": True}, "give `sprites` too"),
    ({"kind": "plate", "sprites": []}, "sprites = []"),
    ({"kind": "plate", "sprites": "a.png", "orient": "sideways"}, "orient = 'sideways'"),
    ({"kind": "plate", "sprites": "a.png", "count": 0}, "count = 0"),
    ({"kind": "plate", "sprites": "a.png", "eyes": 3}, "eyes = 3"),
    ({"kind": "plate", "color": "nope"}, "color"),
    ({"kind": "lens", "count": 0}, "count = 0"),
    ({"kind": "lens", "count": 9}, "count = 9"),
    ({"kind": "lens", "chroma": -1}, "chroma = -1.0"),
    ({"kind": "lens", "tint": "nope"}, "tint"),
])
def test_bad_rings_say_what_is_wrong(ring, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        ins(ring)


def test_a_ring_asks_for_no_layers_of_its_own():
    plain = TR.plan(project([IRIS]), FPS, F0, DAWN)
    for kind in RINGS:
        p = TR.plan(project([{**IRIS, "ring": kind}]), FPS, F0, DAWN)
        assert TR.demands(p) == TR.demands(plain) and TR.needs(p) == TR.needs(plain)
        assert json.loads(json.dumps(p)) == p                              # JSON-clean, as every plan


# ---------------------------------------------------------------- layout and motion
def test_the_parts_of_a_record_and_a_plate_lie_in_order_inside_the_ring():
    v = PT.vinyl_layout(ins("vinyl")["ring"])
    edges = [v["label"][0], v["label"][1], v["wax"][1], v["grooves"][1], v["edge"][1]]
    assert edges[0] == 1.0 and edges[-1] == 2.8 and all(b > a for a, b in zip(edges, edges[1:]))
    assert v["wax"][0] == v["label"][1] and v["grooves"][0] == v["wax"][1] and v["edge"][0] == v["grooves"][1]
    assert all(v["grooves"][0] < t < v["grooves"][1] for t in v["tracks"]) and v["tracks"] == sorted(v["tracks"])
    p = PT.plate_layout(ins({"kind": "plate", "size": 1.6})["ring"])
    assert p["slope"][0] == 1.0 and p["slope"][1] == p["flange"][0] and p["flange"][1] == p["bead"][0]
    assert p["bead"][1] == 1.6 and p["flange"][0] < p["seat"] < p["flange"][1]
    assert all(p["flange"][0] < lo < hi < p["flange"][1] for lo, hi in p["bands"])


@pytest.mark.parametrize("count", range(1, PT.MAX_RINGS + 1))
def test_the_lens_rings_fill_the_ring_one_after_another_and_follow_their_seed(count):
    for seed in range(1, 12):
        ring = ins({"kind": "lens", "count": count, "seed": seed, "size": 2.4})["ring"]
        bands = ring["bands"]
        assert len(bands) == count and bands[0]["a"] == 1.0 and bands[-1]["b"] == 2.4
        assert all(b["a"] < b["b"] for b in bands)
        assert all(x["b"] < y["a"] for x, y in zip(bands, bands[1:]))     # a gap between every two
        assert all(12.0 <= span <= 50.0 for b in bands for _, span in b["breaks"])
        assert [b["turn"] > 0 for b in bands] == [i % 2 == 0 for i in range(count)]     # every other one the other way
        assert ins({"kind": "lens", "count": count, "seed": seed, "size": 2.4})["ring"]["bands"] == bands
    assert ins({"kind": "lens", "seed": 1})["ring"]["bands"] != ins({"kind": "lens", "seed": 2})["ring"]["bands"]


def test_the_record_turns_clockwise_at_its_rpm_and_the_sprites_ride_round_the_rim():
    v = ins("vinyl")["ring"]
    assert PT.spin(v, 0.0) == 0.0 and PT.spin(v, 1.0) == pytest.approx(200.0)    # 33 1/3 turns a minute
    assert PT.spin(ins({"kind": "vinyl", "rpm": -45})["ring"], 1.0) == pytest.approx(-270.0)
    p = ins({"kind": "plate", "sprites": "a.png", "count": 4})["ring"]
    assert PT.seats(p, 0.0) == [(0.0, 0), (90.0, 1), (180.0, 2), (270.0, 3)]   # the first at the top, evenly round
    q = ins({"kind": "plate", "sprites": "a.png", "count": 4, "turn": 30})["ring"]
    assert PT.seats(q, 1.0)[0] == (30.0, 0) and PT.seats(q, 13.0)[0] == (30.0, 0)
    assert sorted(PT.sprite_pick(p, j, 4) for j in range(4)) == [0, 1, 2, 3]   # every one before any comes again


# ---------------------------------------------------------------- compositing
def coded(g, w, h):
    """A picture whose red channel counts the column, blue the row, green says whose it is."""
    img = np.empty((h, w, 3), np.float32)
    img[..., 0] = (np.arange(w, dtype=np.float32) / (w - 1))[None, :]
    img[..., 1] = g
    img[..., 2] = (np.arange(h, dtype=np.float32) / (h - 1))[:, None]
    return img


def flat(colour, w, h):
    return np.broadcast_to(np.asarray(colour, np.float32), (h, w, 3)).copy()


def save_rgb(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))


def build(root, out, ring, spec=IRIS, inner=None, anchor=(0.5, 0.5), frame=None, sprite_of=None, **kw):
    """The plan of one ringed iris (`kw` its own keys), the layers it asks for written to `root` (the plate `inner`, the
    anchor at `anchor` frame fractions) and a CutFx for output `out` (`frame` (w, h) overrides its frame size)."""
    p = TR.plan(project([{**spec, "ring": ring, **kw}]), FPS, F0, DAWN)
    w, h = frame or SIZES[out]
    inner = coded(0.8, w, h) if inner is None else inner
    for f, items in TR.demands(p).items():
        for it in items:
            if it["kind"] == "plate":
                save_rgb(root / TR.plate_rel(it["shot"], f), inner)
            else:
                (root / TR.point_rel(it["key"], f)).parent.mkdir(parents=True, exist_ok=True)
                (root / TR.point_rel(it["key"], f)).write_text(json.dumps({"p": list(anchor), "depth": 2.0, "m": 10.0}))
    return p["inserts"][0], CF.CutFx(p, root, (w, h), out, sprite_of=sprite_of)


def circle(s, f, size, anchor=(0.5, 0.5), out="16x9"):
    return IR.circle(s, IR.state(s, f), out, size, M.frac_to_px(anchor, size), 10.0)


def dist(c, size):
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w]
    return np.hypot(xs - c["centre"][0], ys - c["centre"][1])


def luma(img):
    return img @ np.float32([0.2126, 0.7152, 0.0722])


@pytest.mark.parametrize("out", list(SIZES))
@pytest.mark.parametrize("kind", RINGS)
def test_a_ring_draws_nothing_past_its_reach_and_leaves_the_windows_middle_to_the_iris(tmp_path, out, kind):
    s, fx = build(tmp_path, out, kind, anchor=(0.6, 0.45))
    size = SIZES[out]
    host = coded(0.2, *size)
    for f in range(s["f0"], s["f1"], 3):
        img = fx.frame(f, host)
        c = circle(s, f, size, (0.6, 0.45), out)
        d = dist(c, size)
        far = d >= PORTAL._reach(s["ring"], c["radius"]) + 2.0
        assert np.array_equal(img[far], host[far]), f"frame {f}: drawn past the ring's reach"
        if c["radius"] >= 2.0:
            mid = d <= c["radius"] * (1.0 - s["rim"]["width"]) - 1.0     # inside the vignette: the picture alone
            pic = M.warp_scaled(np.asarray(fx._plate("p", f, host)), c["at"], c["k"])
            assert np.abs(img[mid] - pic[mid]).max() < 2e-3
        assert not np.array_equal(img, host) or c["radius"] < 0.5


@pytest.mark.parametrize("out", list(SIZES))
@pytest.mark.parametrize("kind", RINGS)
def test_an_expand_with_a_ring_still_lands_on_the_inner_picture(tmp_path, out, kind):
    s, fx = build(tmp_path, out, kind, spec=EXPAND)
    host = coded(0.2, *SIZES[out])
    last = fx.frame(s["f1"] - 1, host)
    assert np.abs(last - np.asarray(fx._plate("n", s["f1"] - 1, host))).max() < 1e-6
    before = fx.frame(s["f1"] - 10, host)
    assert not np.array_equal(before, last)


def test_the_record_is_dark_round_a_label_that_turns_clockwise_at_33_and_a_third(tmp_path):
    size = (320, 320)
    s, fx = build(tmp_path, "1x1", "vinyl", inner=flat((0.9, 0.85, 0.8), *size), frame=size, size=0.3)
    host = flat((0.5, 0.6, 0.7), *size)
    f1 = s["f0"] + 12
    f2 = f1 + 5                                                            # 1/6 s later: 33.3 degrees on
    a, b = fx.frame(f1, host), fx.frame(f2, host)
    c = circle(s, f1, size, out="1x1")
    lay = PT.vinyl_layout(s["ring"])
    d = dist(c, size)
    r = c["radius"]
    ys, xs = np.mgrid[0:size[1], 0:size[0]]
    ang = np.degrees(np.arctan2(-(ys - c["centre"][1]), xs - c["centre"][0]))
    off_light = np.abs(((ang - PORTAL.LIGHT_AT + 90.0) % 180.0) - 90.0) > 60.0   # away from the sheen's wedges
    grooves = (d > lay["grooves"][0] * r + 2) & (d < lay["grooves"][1] * r - 2)
    assert float(luma(a[grooves & off_light]).mean()) < 0.15                  # black vinyl
    label = (d > 1.0 * r + 2) & (d < lay["label"][1] * r - 2)
    gold = np.float32(PAL.srgb(DAWN["gold"]))
    assert (np.abs(a[label] - gold).max(axis=1) < 0.12).mean() > 0.4         # mostly its ground
    # round the label, the later frame is the earlier one turned 33.3 degrees clockwise
    def round_label(img, share):
        rr = (1.0 + share * (lay["label"][1] - 1.0)) * r
        a_ = np.radians(np.arange(0.0, 360.0, 0.5))
        xs_ = (c["centre"][0] + rr * np.cos(a_)).astype(np.float32)[None, :]
        ys_ = (c["centre"][1] - rr * np.sin(a_)).astype(np.float32)[None, :]
        return luma(cv2.remap(img, xs_, ys_, cv2.INTER_LINEAR))[0]
    for share in (0.26, 0.68):                                             # the print, the arch and its dot
        pa, pb = round_label(a, share), round_label(b, share)
        pa, pb = pa - pa.mean(), pb - pb.mean()
        corr = [float(np.dot(np.roll(pa, k), pb)) for k in range(len(pa))]
        shift = np.arange(0.0, 360.0, 0.5)[int(np.argmax(corr))]          # degrees counter-clockwise it moved
        assert abs(((shift + PT.spin(s["ring"], 5 / FPS) + 180.0) % 360.0) - 180.0) < 2.0, shift


def test_the_sheen_stands_in_the_light_as_the_record_turns(tmp_path):
    size = (320, 320)
    s, fx = build(tmp_path, "1x1", "vinyl", inner=flat((0.9, 0.85, 0.8), *size), frame=size, size=0.3)
    host = flat((0.5, 0.6, 0.7), *size)
    for f in (s["f0"] + 10, s["f0"] + 17, s["f0"] + 31):
        img = fx.frame(f, host)
        c = circle(s, f, size, out="1x1")
        lay = PT.vinyl_layout(s["ring"])
        mid = (lay["grooves"][0] + lay["grooves"][1]) / 2.0 * c["radius"]
        angles = np.arange(0.0, 360.0, 2.0)
        vals = []
        for a in angles:
            x = c["centre"][0] + mid * math.cos(math.radians(a))
            y = c["centre"][1] - mid * math.sin(math.radians(a))
            vals.append(float(luma(img[int(round(y)), int(round(x))])))
        top = angles[int(np.argmax(vals))]
        off = min(abs((top - PORTAL.LIGHT_AT + 180.0) % 360.0 - 180.0), abs((top - PORTAL.LIGHT_AT) % 360.0 - 180.0))
        assert off < 25.0, f"frame {f}: the sheen at {top} degrees"


def sprite(path, colour=(40, 200, 60)):
    """A round sprite with a clear ground, written to `path`."""
    img = np.zeros((64, 64, 4), np.uint8)
    cv2.circle(img, (32, 32), 28, (*colour[::-1], 255), -1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def test_the_plate_is_lit_from_the_upper_left_and_sprites_stand_on_its_rim(tmp_path):
    size = (320, 320)
    host = flat((0.3, 0.3, 0.35), *size)
    s, fx = build(tmp_path / "bare", "1x1", {"kind": "plate", "band": False}, frame=size, size=0.45)
    img = fx.frame(s["f0"] + 12, host)
    c = circle(s, s["f0"] + 12, size, out="1x1")
    lay = PT.plate_layout(s["ring"])
    d = dist(c, size)
    ys, xs = np.mgrid[0:size[1], 0:size[0]]
    rim = (d > lay["flange"][0] * c["radius"] + 1) & (d < lay["flange"][1] * c["radius"] - 1)
    ul = rim & (xs < c["centre"][0]) & (ys < c["centre"][1])
    lr = rim & (xs > c["centre"][0]) & (ys > c["centre"][1])
    assert float(luma(img[ul]).mean()) > float(luma(img[lr]).mean()) + 0.02   # the glaze highlight at the upper left
    assert float(luma(img[rim]).mean()) > 0.8                                 # a light glaze
    red = sprite(tmp_path / "s" / "dot.png", (230, 30, 30))
    s2, fx2 = build(tmp_path / "sprites", "1x1", {"kind": "plate", "band": False, "sprites": str(red), "count": 4},
                    frame=size, size=0.45)
    img2 = fx2.frame(s2["f0"] + 12, host)
    top = (int(round(c["centre"][0])), int(round(c["centre"][1] - lay["seat"] * c["radius"])))
    assert img2[top[1], top[0], 0] > 0.7 and img2[top[1], top[0], 1] < 0.3    # the first sprite at the top of the rim
    assert img[top[1], top[0], 0] > 0.7 and img[top[1], top[0], 1] > 0.7      # glaze there without it


def test_googly_eyes_on_the_plates_sprites(tmp_path):
    size = (320, 320)
    host = flat((0.3, 0.3, 0.35), *size)
    green = sprite(tmp_path / "s" / "pea.png")
    ring = {"kind": "plate", "band": False, "sprites": str(green), "count": 4, "scale": 0.6}
    s, fx = build(tmp_path / "plain", "1x1", ring, frame=size, size=0.45)
    s2, fx2 = build(tmp_path / "eyes", "1x1", {**ring, "eyes": True}, frame=size, size=0.45)
    a, b = fx.frame(s["f0"] + 12, host), fx2.frame(s2["f0"] + 12, host)
    changed = np.abs(a - b).max(axis=2) > 0.2
    assert changed.sum() > 20                                                 # whites and pupils on the sprites
    c = circle(s, s["f0"] + 12, size, out="1x1")
    d = dist(c, size)
    assert (d[changed] > c["radius"]).all()                                   # on the rim, not in the window
    assert float(luma(b[changed]).max()) > 0.85 and float(luma(b[changed]).min()) < 0.2     # white and black


def test_the_lens_rings_show_the_inner_picture_with_the_host_between_them(tmp_path):
    size = (320, 320)
    host = flat((0.1, 0.15, 0.85), *size)                                     # blue
    s, fx = build(tmp_path, "1x1", {"kind": "lens", "count": 3, "seed": 4}, inner=flat((0.1, 0.85, 0.15), *size),
                  frame=size, size=0.3)
    img = fx.frame(s["f0"] + 12, host)
    c = circle(s, s["f0"] + 12, size, out="1x1")
    d = dist(c, size) / c["radius"]
    bands = s["ring"]["bands"]
    on = np.zeros(d.shape, bool)
    for b in bands:
        on |= (d > b["a"] + 0.05) & (d < b["b"] - 0.05)
    gaps = np.zeros(d.shape, bool)
    for x, y in zip(bands, bands[1:]):
        gaps |= (d > x["b"] + 0.03) & (d < y["a"] - 0.03)
    greenish = img[..., 1] > img[..., 2]
    assert greenish[on].mean() > 0.5                                          # the inner world through the tubes
    assert (~greenish[gaps]).mean() > 0.95                                    # the host (in the glow) between them
    s2, fx2 = build(tmp_path / "split", "1x1", {"kind": "lens", "count": 3, "seed": 4, "chroma": 30},
                    inner=coded(0.5, *size), frame=size, size=0.3)
    s3, fx3 = build(tmp_path / "plain", "1x1", {"kind": "lens", "count": 3, "seed": 4, "chroma": 0},
                    inner=coded(0.5, *size), frame=size, size=0.3)
    split, plain = fx2.frame(s2["f0"] + 12, host), fx3.frame(s3["f0"] + 12, host)
    assert np.abs(split[on][:, 1] - plain[on][:, 1]).max() < 1e-4             # green stays put
    assert np.abs(split[on][:, 0] - plain[on][:, 0]).max() > 0.02             # red and blue move apart


def test_a_plate_whose_sprites_find_nothing_is_an_error_naming_the_insert(tmp_path):
    def nothing(refs):
        raise FX.FxError("sprites: no picture in the asset registry has the tags ['veggie']")
    s, fx = build(tmp_path, "1x1", {"kind": "plate", "sprites": {"tag": "veggie"}}, sprite_of=nothing)
    with pytest.raises(TR.TransitionError, match=re.escape("insert 0 (iris 1.0-3.0 s): ring: sprites: no picture")):
        fx.check([s["f0"]])
