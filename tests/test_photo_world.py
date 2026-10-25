"""The photo_world set's maths (mkmmd.core.photoworld): the photo's frame (axis, up, roll), where a world direction falls on
the photo (a pinhole at spread = fov, the angles stretched about the centre at a larger spread, aspect-correct, the fill
behind and outside it), the inverse, the spec parser and the image's file or registry slug; then a real build (skipped
without Blender). The node tree is checked against these numbers by rendering a numbered grid (docs/design.md: Sets)."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN
from mkmmd.core import photoworld as PW


def cfg(**kw):
    return PW.parse({"image": "street.jpg", "fov": 60.0, **kw})


def angle(a, b):
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    c = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# ---------------------------------------------------------------------------------------------------------------- frame
@pytest.mark.parametrize("toward,roll", [([0, 1, 0], 0), ([1, 1, 0.4], 20), ([-0.3, 0.2, -0.9], -75), ([0, 0, 1], 33)])
def test_the_frame_is_orthonormal_and_right_handed(toward, roll):
    f, r, u = PW.frame(toward, roll)
    for v in (f, r, u):
        assert np.linalg.norm(v) == pytest.approx(1.0)
    assert f @ r == pytest.approx(0.0, abs=1e-12) and f @ u == pytest.approx(0.0, abs=1e-12)
    assert r @ u == pytest.approx(0.0, abs=1e-12)
    assert np.cross(f, u) == pytest.approx(r)                            # right = axis x up, rolled or not
    assert f == pytest.approx(np.asarray(toward, float) / np.linalg.norm(toward))


def test_unrolled_the_photo_is_level_with_its_up_toward_the_sky():
    f, r, u = PW.frame([0, 1, 0])
    assert r == pytest.approx([1, 0, 0]) and u == pytest.approx([0, 0, 1])     # looking along +Y: right +X, up +Z
    f, r, u = PW.frame([1, -2, 0.7])
    assert r[2] == pytest.approx(0.0, abs=1e-12) and u[2] > 0.0
    f, r, u = PW.frame([0, 0, -1])                                       # straight down: +Y is the photo's up
    assert u == pytest.approx([0, 1, 0]) and r == pytest.approx([1, 0, 0])


def test_roll_turns_the_photo_clockwise_as_seen_along_toward():
    c0, c90 = cfg(toward=[0, 1, 0]), cfg(toward=[0, 1, 0], roll=90)
    top = (0.5, 1.0)                                                     # the middle of the photo's top edge
    assert PW.direction(*top, c0, (1500, 1000))[0][2] > 0.3                # unrolled: above the centre
    d = PW.direction(*top, c90, (1500, 1000))[0]
    assert d[0] > 0.3 and abs(d[2]) < 1e-9                               # turned clockwise: the top is on the right


# ---------------------------------------------------------------------------------------------------------------- mapping
def test_the_photo_s_centre_is_toward_and_its_long_side_spans_fov():
    c = cfg(toward=[1, 2, 0.5], roll=12)
    f = PW.frame(c["toward"])[0]
    assert PW.direction(0.5, 0.5, c, (1500, 1000))[0] == pytest.approx(f)
    s, t, inside = PW.uv(f, c, (1500, 1000))
    assert (s[0], t[0], inside[0]) == (pytest.approx(0.5), pytest.approx(0.5), True)
    for size in ((1500, 1000), (1000, 1500)):                            # landscape and portrait: the long side is fov
        long_edge = (1.0, 0.5) if size[0] > size[1] else (0.5, 1.0)
        short_edge = (0.5, 1.0) if size[0] > size[1] else (1.0, 0.5)
        assert angle(PW.direction(*long_edge, c, size), f)[0] == pytest.approx(30.0)
        assert angle(PW.direction(*short_edge, c, size), f)[0] == pytest.approx(
            math.degrees(math.atan(math.tan(math.radians(30.0)) / 1.5)))       # aspect-correct: 21.05 degrees
        assert angle(PW.direction(1.0, 1.0, c, size), f)[0] == pytest.approx(
            math.degrees(math.atan(math.tan(math.radians(30.0)) * math.hypot(1.0, 1.0 / 1.5))))


def test_spread_equal_to_fov_is_a_pinhole_camera():
    c = cfg(toward=[0.2, 1.0, -0.1], roll=8, fov=70.0)
    f, r, u = PW.frame(c["toward"], c["roll"])
    rng = np.random.default_rng(4)
    d = f + rng.uniform(-0.4, 0.4, (200, 3))
    x, y, seen = PW.photo_xy(d, c)
    k = math.tan(math.radians(35.0))
    assert seen.all()
    assert x == pytest.approx((d @ r) / (d @ f) / k) and y == pytest.approx((d @ u) / (d @ f) / k)
    line = np.array([1.0, 3.0, 0.4]) + np.linspace(-2, 2, 9)[:, None] * np.array([0.5, 0.2, 0.3])   # a straight edge
    x, y, _ = PW.photo_xy(line, c)
    a, b = np.stack([x - x[0], y - y[0]], 1), np.array([x[-1] - x[0], y[-1] - y[0]])
    assert np.abs(a[:, 0] * b[1] - a[:, 1] * b[0]).max() < 1e-9          # stays straight on the photo


def test_a_wider_spread_stretches_angles_uniformly_about_the_centre():
    rng = np.random.default_rng(5)
    st = rng.uniform(0.02, 0.98, (100, 2))
    a, b = cfg(toward=[0, 1, 0.3], roll=10), cfg(toward=[0, 1, 0.3], roll=10, spread=150.0)
    f = PW.frame(a["toward"])[0]
    da, db = PW.direction(st[:, 0], st[:, 1], a, (1500, 1000)), PW.direction(st[:, 0], st[:, 1], b, (1500, 1000))
    assert angle(db, f) == pytest.approx(angle(da, f) * 150.0 / 60.0)     # theta = alpha * spread / fov
    side_a, side_b = da - (da @ f)[:, None] * f, db - (db @ f)[:, None] * f
    assert angle(side_a, side_b) == pytest.approx(np.zeros(100), abs=1e-4)     # the same azimuth (arccos's precision)
    assert angle(PW.direction(1.0, 0.5, b, (1500, 1000)), f)[0] == pytest.approx(75.0)   # the long side spans 150


def test_behind_the_photo_and_outside_its_rectangle_is_fill():
    c = cfg(toward=[0, 1, 0])
    s, t, inside = PW.uv(np.array([[0, -1, 0], [1, 0.01, 0], [0, 1, 0.9], [0.5, 1, 0.0], [0.6, 1, 0]]), c, (1500, 1000))
    assert inside.tolist() == [False, False, False, True, False]         # behind, 89 deg aside, above, in, past the edge
    s, t, inside = PW.uv(np.array([[0.6, 1, 0]]), cfg(toward=[0, 1, 0], spread=70.0), (1500, 1000))
    assert inside[0]                                                     # a little more spread reaches it
    x, y, seen = PW.photo_xy(np.array([[1, 0.2, 0], [0, -1, 0.1]]), cfg(toward=[0, 1, 0], fov=120.0, spread=100.0))
    assert not seen.any()                                                # spread < fov: alpha reaches 90 before theta


@pytest.mark.parametrize("spread", [40.0, 60.0, 200.0, 300.0])
def test_uv_and_direction_are_each_other_s_inverse(spread):
    c = cfg(toward=[-1, 0.5, 0.2], roll=-30, spread=spread)
    rng = np.random.default_rng(6)
    st = rng.uniform(0.01, 0.99, (300, 2))
    for size in ((1500, 1000), (900, 1600)):
        d = PW.direction(st[:, 0], st[:, 1], c, size)
        s, t, inside = PW.uv(d * 3.0, c, size)                           # any length of d will do
        keep = angle(d, PW.frame(c["toward"])[0]) < 179.0                # past the back pole a point has no azimuth
        assert inside[keep].all() and s[keep] == pytest.approx(st[keep, 0]) and t[keep] == pytest.approx(st[keep, 1])


# ---------------------------------------------------------------------------------------------------------------- spec
def test_parse_fills_defaults():
    c = PW.parse({"name": "street", "kind": "photo_world", "at": [0, 0, 0], "yaw": 30, "image": "x.jpg", "fov": 55})
    assert c["spread"] == 55 and c["light"] == c["strength"] == 1.0 and c["toward"] == [0.0, 1.0, 0.0]
    assert c["roll"] == 0.0 and c["saturation"] == 1.0 and c["fill"] == "#080808"
    c = PW.parse({"image": "x.jpg", "fov": 55, "toward": [0, 3, 4], "strength": 0.8, "light": 6})
    assert c["toward"] == pytest.approx([0.0, 0.6, 0.8]) and (c["strength"], c["light"]) == (0.8, 6)
    assert PW.SPEC["toward"] == [0.0, 1.0, 0.0] and PW.SPEC["spread"] is None      # the defaults are not modified


@pytest.mark.parametrize("spec,frag", [
    ({"fov": 60}, "give `image`"), ({"image": "x.jpg"}, "give `fov`"), ({"image": 3, "fov": 60}, "image is"),
    ({"image": "x.jpg", "fov": 60, "colour": "base"}, "unknown keys"), ({"image": "x.jpg", "fov": 0}, "fov is"),
    ({"image": "x.jpg", "fov": 180}, "fov is"), ({"image": "x.jpg", "fov": 60, "spread": 400}, "spread is"),
    ({"image": "x.jpg", "fov": 60, "spread": 0}, "spread is"), ({"image": "x.jpg", "fov": 60, "toward": [0, 0, 0]}, "toward"),
    ({"image": "x.jpg", "fov": 60, "toward": [1, 2]}, "toward"), ({"image": "x.jpg", "fov": 60, "strength": -1}, "strength"),
    ({"image": "x.jpg", "fov": 60, "light": "bright"}, "light"), ({"image": "x.jpg", "fov": 60, "saturation": 11}, "saturation"),
    ({"image": "x.jpg", "fov": 60, "roll": True}, "roll"), ({"image": "x.jpg", "fov": 60, "fill": 3}, "fill"),
])
def test_parse_rejects_bad_specs(spec, frag):
    with pytest.raises(PW.PhotoWorldError, match=frag):
        PW.parse(spec)


def test_the_image_is_a_file_or_a_registry_slug(tmp_path):
    assert PW.is_file("plates/a.jpg") and PW.is_file("street.webp") and PW.is_file("~/x") and not PW.is_file("pexels_1")
    (tmp_path / "proj" / "plates").mkdir(parents=True)
    Image.new("RGB", (4, 3)).save(tmp_path / "proj" / "plates" / "a.jpg")
    assert PW.resolve("plates/a.jpg", tmp_path / "proj") == tmp_path / "proj" / "plates" / "a.jpg"
    assert PW.resolve(str(tmp_path / "proj" / "plates" / "a.jpg")) == tmp_path / "proj" / "plates" / "a.jpg"
    with pytest.raises(PW.PhotoWorldError, match="does not exist"):
        PW.resolve("plates/b.jpg", tmp_path / "proj")
    with pytest.raises(PW.PhotoWorldError, match="is not an image"):
        PW.resolve("plates/a.mp4", tmp_path / "proj")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "registry.json").write_text(json.dumps([
        {"slug": "plate", "kind": "reference", "path": str(tmp_path / "proj" / "plates" / "a.jpg")},
        {"slug": "rel", "kind": "texture", "path": "textures/r.png"},
        {"slug": "clip", "kind": "reference", "path": str(tmp_path / "c.mp4")}]), encoding="utf-8")
    assert PW.resolve("plate", None, assets) == tmp_path / "proj" / "plates" / "a.jpg"
    with pytest.raises(PW.PhotoWorldError, match="asset's file .*r.png does not exist"):
        PW.resolve("rel", None, assets)                                  # a relative path is the registry's
    with pytest.raises(PW.PhotoWorldError, match="is not an image"):
        PW.resolve("clip", None, assets)
    with pytest.raises(PW.PhotoWorldError, match="the images there: plate, rel"):
        PW.resolve("street", None, assets)


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
duration = 1.0
blend = "build/t.blend"

[[set]]
name = "street"
kind = "photo_world"
image = "{image}"
toward = [0.0, 1.0, 0.2]
fov = 60
spread = 120
light = 3
fill = "pine"

[[prop]]
name = "ground"
card = "library:floor"
slots = {{ kind = "asphalt", fade = [4, 10], fade_to = "clear", puddles = 0.2, wet = 0.5 }}
"""


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_a_photo_world_builds_the_scene_s_world_and_a_missing_or_unreadable_photo_says_so(tmp_path, capsys):
    (tmp_path / "plates").mkdir()
    Image.new("RGB", (30, 20), (200, 120, 40)).save(tmp_path / "plates" / "street.png")
    (tmp_path / "mk.toml").write_text(SCENE.format(image="plates/street.png"), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        MAIN.main(["build", "--project", str(tmp_path), "--until", "props"])
    out = json.loads(capsys.readouterr().out)
    assert e.value.code == 0, out.get("error")
    assert out["stages"]["sets"]["street"]["kind"] == "photo_world"
    expr = ("[bpy.context.scene.world.name, bpy.context.scene.world.sun_threshold, "
            "[n.image.size[:] for n in bpy.context.scene.world.node_tree.nodes if n.type == 'TEX_IMAGE'], "
            "sorted({n.type for n in bpy.context.scene.world.node_tree.nodes} & {'LIGHT_PATH', 'BACKGROUND'}), "
            "bpy.data.materials['ground_floor'].surface_render_method, bpy.context.scene.eevee.use_raytracing]")
    with pytest.raises(SystemExit):
        MAIN.main(["q", str(tmp_path / "build" / "t.blend"), expr, "--frames", "31", "--project", str(tmp_path)])
    (row,) = json.loads(capsys.readouterr().out)["values"]
    assert row[0] == "street" and row[1] == 0.0 and row[2] == [[30, 20]] and row[3] == ["BACKGROUND", "LIGHT_PATH"]
    assert row[4] == "DITHERED" and row[5] is True                      # the floor fades to clear; EEVEE traces rays
    (tmp_path / "plates" / "bad.png").write_bytes(b"not a png at all")
    for image, says in (("plates/nope.png", "does not exist"), ("plates/bad.png", "no pixels Blender can read")):
        (tmp_path / "mk.toml").write_text(SCENE.format(image=image), encoding="utf-8")
        with pytest.raises(SystemExit) as e:
            MAIN.main(["build", "--project", str(tmp_path), "--until", "sets"])
        out = json.loads(capsys.readouterr().out)
        assert e.value.code == 3 and "photo_world" in out["error"] and says in out["error"], out["error"]
