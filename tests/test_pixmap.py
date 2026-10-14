"""Pixel maps (mkmmd.pixmap: mk look --map, mk measure) and marks (mkmmd.marks: mk mark): the nearest surface at each
pixel through Blender's camera, measurements read off it, marks drawn where they belong. The last test renders a small
scene in Blender and checks the map against the picture (skipped without Blender)."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_geom_checks import BLENDER_SHIFT_UV

from mkmmd import bridge
from mkmmd import config as CFG
from mkmmd import marks as MK
from mkmmd import pixmap as PX
from mkmmd.cli import main as MAIN

ROOT = Path(__file__).resolve().parent.parent


def look_at(loc, target, w=200, h=150, lens=50.0, **extra):
    """A camera at `loc` looking at `target` (z up), as mkmmd.pixmap.Camera reads it."""
    loc, target = np.asarray(loc, float), np.asarray(target, float)
    back = (loc - target) / np.linalg.norm(loc - target)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, np.cross(back, right), back, loc
    info = {"matrix": M.tolist(), "type": "PERSP", "lens": lens, "sensor": [36.0, 24.0], "fit": "AUTO", "shift": [0, 0],
            "ortho_scale": 6.0, "clip_start": 0.01, "size": [w, h]}
    info.update(extra)
    return PX.Camera(info)


def quads(*specs):
    """Geometry of upright rectangles facing -y: (x0, x1, z0, z1, y, material) each, one object per material."""
    P, T, mats = [], [], []
    for x0, x1, z0, z1, y, mat in specs:
        o = len(P)
        P += [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
        T += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        mats += [mat, mat]
    names = sorted(set(mats))
    tm = np.array([names.index(m) for m in mats], np.int32)
    P = np.asarray(P, float)
    return {"posed": P, "rest": P, "tris": np.asarray(T), "tri_object": tm, "tri_material": tm,
            "tri_face": np.arange(len(T)) // 2, "objects": np.array(names), "materials": np.array(names)}


def test_the_nearest_surface_owns_each_pixel_whatever_the_order():
    """Two overlapping squares, a near one in front of a far one: where both are, the near one shows, given in either
    order (the depth test, not the drawing order, decides)."""
    cam = look_at((0, -4, 1), (0, 0, 1))
    near_first = quads((-0.5, 0.5, 0.5, 1.5, -1.0, "near"), (-1.0, 1.0, 0.0, 2.0, 1.0, "far"))
    far_first = quads((-1.0, 1.0, 0.0, 2.0, 1.0, "far"), (-0.5, 0.5, 0.5, 1.5, -1.0, "near"))
    for geo in (near_first, far_first):
        m = PX.Map.make(geo, cam)
        x, y, _ = cam.project(np.array([[0.0, -1.0, 1.0]]))
        assert m.at(int(x[0]), int(y[0]))["material"] == "near"
        assert m.at(int(x[0]), int(y[0]))["depth"] == pytest.approx(3.0, abs=1e-4)
        x, y, _ = cam.project(np.array([[0.9, 1.0, 1.9]]))                     # the far square beyond the near one
        assert m.at(int(x[0]), int(y[0]))["material"] == "far"


def test_a_floor_that_runs_behind_the_camera_still_covers_the_picture_below_the_horizon():
    """A floor that reaches behind the camera has corners with no place on the picture; it still covers every pixel
    whose ray goes down to it in front of the camera, and nothing above the horizon."""
    cam = look_at((0, -4, 1), (0, 0, 1))
    P = np.array([[-1000, -1000, 0], [1000, -1000, 0], [1000, 1000, 0], [-1000, 1000, 0]], float)
    m = PX.Map.make({"posed": P, "rest": P, "tris": np.array([[0, 1, 2], [0, 2, 3]]), "tri_object": np.zeros(2, np.int32),
                     "tri_material": np.zeros(2, np.int32), "tri_face": np.zeros(2, np.int32),
                     "objects": np.array(["floor"]), "materials": np.array(["floor"])}, cam)
    floor = m.mask("floor")
    assert floor[75:].all() and not floor[:75].any()                       # the horizon runs through the middle
    p = m.points([100], [120])[0]
    assert p[2] == pytest.approx(0.0, abs=1e-6)
    assert m.depth[120, 100] == pytest.approx(1.0 / ((120.5 - 75.0) / cam.f), rel=1e-6)


def test_the_surface_point_of_a_pixel_lies_on_its_ray():
    """The point a pixel shows lies on the surface where the pixel's centre ray meets it: back through the camera it
    lands on that centre, even on a large surface seen at a slant (positions interpolated across the picture instead of
    along the rays would miss by pixels)."""
    cam = look_at((0, -3, 1), (0, 0, 1))
    P = np.array([[-2, -1, 0], [2, -1, 0], [0, 6, 3]], float)
    m = PX.Map.make({"posed": P, "rest": P, "tris": np.array([[0, 1, 2]]), "tri_object": np.zeros(1, np.int32),
                     "tri_material": np.zeros(1, np.int32), "tri_face": np.zeros(1, np.int32),
                     "objects": np.array(["slope"]), "materials": np.array(["slope"])}, cam)
    ys, xs = np.nonzero(m.tri >= 0)
    assert len(xs) > 5000
    W = m.points(xs, ys)
    x, y, _ = cam.project(W)
    assert np.abs(x - (xs + 0.5)).max() < 1e-3 and np.abs(y - (ys + 0.5)).max() < 1e-3
    n = np.cross(P[1] - P[0], P[2] - P[0])
    assert np.abs((W - P[0]) @ n / np.linalg.norm(n)).max() < 1e-6


@pytest.mark.parametrize("size", sorted(BLENDER_SHIFT_UV))
def test_the_camera_places_points_as_blender_does(size):
    """Lens shift and sensor fit as Blender's own world_to_camera_view puts them (the probed values of
    test_geom_checks), here in pixels from the top left."""
    w, h = size
    for shift, (u, v) in BLENDER_SHIFT_UV[size].items():
        cam = PX.Camera({"matrix": np.eye(4).tolist(), "lens": 50.0, "sensor": [36.0, 24.0], "fit": "AUTO",
                         "shift": list(shift), "size": [w, h]})
        x, y, d = cam.project(np.array([[0.144 * 36.0 / 50.0, 0.086 * 36.0 / 50.0, -1.0]]))
        assert d[0] == pytest.approx(1.0)
        assert (x[0], y[0]) == pytest.approx((u * w, (1.0 - v) * h), abs=1e-3), (size, shift)


def test_edges_extents_and_gaps_are_read_off_the_map():
    """Two upright slabs with a gap between them and a strip across both, in front: a strip's top and bottom per column
    (its height as seen and in the world), a slab's outer sides per row, and the gap between the slabs (not the open
    space outside them)."""
    cam = look_at((0, -4, 1), (0, 0, 1), w=400, h=300)
    m = PX.Map.make(quads((-0.6, -0.2, 0.0, 2.0, 0.0, "slab"), (0.2, 0.6, 0.0, 2.0, 0.0, "slab"),
                          (-0.6, 0.6, 1.2, 1.4, -0.01, "strip")), cam)
    x, y, _ = cam.project(np.array([[0.4, -0.01, 1.4], [0.4, -0.01, 1.2]]))
    col = int(x[0])
    e = m.edges("strip", [col])[0]
    assert e["top"] == int(np.ceil(y[0] - 0.5)) and e["bottom"] == int(np.floor(y[1] - 0.5))
    assert e["runs"] == 1
    assert e["seen_cm"] == pytest.approx(20.0, abs=m.cm_per_px([col], [e["top"]])[0] * 1.5)
    assert e["cm"] == pytest.approx(20.0, abs=1.0) and e["cm_rest"] == e["cm"]
    _, yr, _ = cam.project(np.array([[0.0, 0.0, 0.6]]))
    row = int(yr[0])
    xs, _, _ = cam.project(np.array([[-0.6, 0, 0.6], [-0.2, 0, 0.6], [0.2, 0, 0.6], [0.6, 0, 0.6]]))
    ext = m.extent("slab", [row])[0]
    assert (ext["left"], ext["right"]) == (int(np.ceil(xs[0] - 0.5)), int(np.floor(xs[3] - 0.5)))
    assert ext["runs"] == 2
    g = m.gaps([row])[0]["gaps"]
    assert len(g) == 1                                                       # outside the slabs is no gap
    assert (g[0]["left"], g[0]["right"]) == (int(np.floor(xs[1] - 0.5)) + 1, int(np.ceil(xs[2] - 0.5)) - 1)
    assert g[0]["cm"] == pytest.approx(40.0, abs=1.5)
    with pytest.raises(PX.MapError, match="it has: slab, strip"):
        m.mask("knit")


def test_marks_given_in_world_points_land_where_the_map_projects_them():
    """A mark at a world point (metres) is drawn at the pixel the image's camera puts that point on, the same as the
    mark given there in pixels."""
    cam = look_at((0, -4, 1), (0, 0, 1), w=200, h=150)
    pt = [0.3, 0.0, 1.3]
    x, y, _ = cam.project(np.array([pt]))
    im = Image.new("RGB", (200, 150), (255, 255, 255))
    world = MK.draw(im, MK.check({"circle": [{"at": pt, "radius": 6, "color": "#ff0000"}]}),
                    MK.View(project=cam.project))
    pixel = MK.draw(im, MK.check({"circle": [{"at": [float(x[0]), float(y[0])], "radius": 6, "color": "#ff0000"}]}))
    assert np.array_equal(np.asarray(world), np.asarray(pixel))
    red = np.argwhere(np.asarray(world)[..., 1] < 100)
    assert abs(red[:, 1].mean() - x[0]) < 1.0 and abs(red[:, 0].mean() - y[0]) < 1.0
    with pytest.raises(MK.MarkError, match="pixel map"):
        MK.draw(im, MK.check({"circle": [{"at": pt, "radius": 6}]}))


def test_a_crop_and_a_zoom_keep_the_images_own_coordinates():
    """Marks and the grid are given in the image's pixels: on a crop from (30, 30) zoomed twice, a box from x = 40 is
    drawn at 20 and the grid line at x = 50 at 40."""
    im = Image.new("RGB", (100, 100), (255, 255, 255))
    pic, view = MK.crop(im, (30, 30, 80, 80), 2.0)
    assert pic.size == (100, 100)
    out = np.asarray(MK.draw(pic, MK.check({"box": [{"box": [40, 40, 10, 10], "color": "#ff0000", "width": 1}]}), view))
    assert tuple(out[30, 20]) == (255, 0, 0) and tuple(out[30, 18]) == (255, 255, 255)
    g = np.asarray(MK.grid(pic, view, step=10))
    assert (g[50, 40] != 255).any() and (g[50, 38] == 255).all()        # row 50 is y = 55: between grid lines


def test_a_fill_between_two_lines_covers_the_space_between_them():
    im = Image.new("RGB", (100, 100), (255, 255, 255))
    marks = MK.check({"line": [{"name": "top", "points": [[10, 20], [90, 20]], "width": 1},
                               {"name": "bottom", "points": [[10, 60], [90, 60]], "width": 1}],
                      "fill": [{"between": ["top", "bottom"], "color": "#0000ff", "opacity": 1.0}]})
    out = np.asarray(MK.draw(im, marks))
    assert tuple(out[40, 30]) == (0, 0, 255) and tuple(out[40, 70]) == (0, 0, 255)
    assert tuple(out[10, 50]) == (255, 255, 255) and tuple(out[70, 50]) == (255, 255, 255)


def test_a_misspelt_mark_is_an_error_that_names_it():
    with pytest.raises(MK.MarkError, match="unknown colour"):
        MK.check({"line": [{"points": [[0, 0], [1, 1]], "colour": "red"}]})
    with pytest.raises(MK.MarkError, match="unknown kind"):
        MK.check({"lines": [{"points": [[0, 0], [1, 1]]}]})
    with pytest.raises(MK.MarkError, match="no \\[\\[line\\]\\] has"):
        MK.check({"fill": [{"between": ["a", "b"]}]})


# ---------------------------------------------------------------- in Blender
SCENE = r'''
import bpy
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.eevee.taa_render_samples = 1
sc.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("w")
sc.world = world

def emit(name, rgb):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs[0])
    return m

def mesh(name, verts, faces, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    ob.data.materials.append(emit(name, mat))
    return ob

# a red wall behind, a green box in front off to the side (where a camera mismatch moves things most), a blue strip bent
# by a bone
mesh("red", [(-9, 4, -9), (9, 4, -9), (9, 4, 9), (-9, 4, 9)], [(0, 1, 2, 3)], (1, 0, 0))
box = [(x, y, z) for x in (1.2, 1.8) for y in (-0.3, 0.3) for z in (0.3, 0.9)]
mesh("green", box, [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)], (0, 1, 0))
strip = [(x / 10.0, -1.0, z) for x in range(0, 11) for z in (1.2, 1.4)]
blue = mesh("blue", strip, [(2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1) for i in range(10)], (0, 0, 1))
ad = bpy.data.armatures.new("rig")
arm = bpy.data.objects.new("rig", ad)
sc.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
b = ad.edit_bones.new("b")
b.head, b.tail = (0, -1, 1.3), (1, -1, 1.3)
bpy.ops.object.mode_set(mode="POSE")
arm.pose.bones["b"].rotation_mode = "XYZ"
arm.pose.bones["b"].rotation_euler = (0, 0.5, 0)
bpy.ops.object.mode_set(mode="OBJECT")
g = blue.vertex_groups.new(name="b")
g.add(list(range(len(strip))), 1.0, "REPLACE")
mod = blue.modifiers.new("arm", "ARMATURE")
mod.object = arm
cd = bpy.data.cameras.new("Cam")
cd.lens = 35.0
cam = bpy.data.objects.new("Cam", cd)
sc.collection.objects.link(cam)
cam.location = (0.3, -6.0, 1.2)
cam.rotation_euler = (1.5707963, 0, 0)
sc.camera = cam
sc.render.resolution_x, sc.render.resolution_y = 160, 120
bpy.ops.wm.save_as_mainfile(filepath=OUT)
'''


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_the_map_of_a_render_shows_what_the_render_shows(tmp_path, capsys):
    """mk look --map on a real render, through a named camera and an orbit view: the material the map gives each pixel
    is the colour the render shows there (away from edges, where the render blends neighbours), each shape's box is the
    same to a pixel in both (a camera a little off moves the box off the centre by more), and a mesh bent by its
    armature is mapped where it renders, with its rest points where its mesh has them."""
    blend = tmp_path / "scene.blend"
    r = subprocess.run([CFG.load()["blender"], "-b", str(bridge.empty_blend(CFG.load())), "-y", "--python-exit-code", "3",
                        "--python-expr", f"OUT = {str(blend)!r}\n" + SCENE], capture_output=True, text=True, timeout=300,
                       check=False)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    with pytest.raises(SystemExit) as e:
        MAIN.main(["look", str(blend), "--frames", "1", "--cam", "Cam", "--view", "20:10", "--target", "(0, 0, 0.9)",
                   "--dist", "5", "--size", "160", "--map", "--out", str(tmp_path / "look")])
    res = json.loads(capsys.readouterr().out)
    assert e.value.code == 0, res
    assert len(res["maps"]) == 2 and not list((tmp_path / "look").glob(".map_geo_*"))
    for img, mp in zip(res["images"], res["maps"]):
        m = PX.Map.load(mp)
        rgb = np.asarray(Image.open(img).convert("RGB"), int)
        seen = np.argmax(rgb, axis=2)                                        # 0 red, 1 green, 2 blue
        mat = m.material()
        want = np.array([m.materials.index(n) if n in m.materials else -1 for n in ("red", "green", "blue")])
        inner = np.ones_like(mat, bool)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            inner &= np.roll(mat, (dy, dx), (0, 1)) == mat
        inner[[0, -1]] = inner[:, [0, -1]] = False
        agree = (want[seen] == mat)[inner].mean()
        assert agree > 0.995, (img, agree)
        for k, name in ((1, "green"), (2, "blue")):
            ys, xs = np.nonzero(mat == m.materials.index(name))
            ry, rx = np.nonzero((seen == k) & (rgb[..., k] > 128))
            assert len(xs) > 50, (img, name)
            box_map = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            box_seen = np.array([rx.min(), ry.min(), rx.max(), ry.max()])
            assert np.abs(box_map - box_seen).max() <= 1, (img, name, box_map, box_seen)
        ys, xs = np.nonzero(mat == m.materials.index("blue"))
        world, rest = m.points(xs, ys), m.points(xs, ys, rest=True)
        assert np.abs(rest[:, 1] + 1.0).max() < 1e-4                       # the strip's own mesh: the plane y = -1
        # turned 0.5 rad about its bone (the line y = -1, z = 1.3): along it nothing moves, round it the distance stays
        assert np.abs(world[:, 0] - rest[:, 0]).max() < 1e-4
        r_w, r_r = np.hypot(world[:, 1] + 1.0, world[:, 2] - 1.3), np.hypot(rest[:, 1] + 1.0, rest[:, 2] - 1.3)
        assert np.abs(r_w - r_r).max() < 1e-4 and np.abs(world[:, 1] - rest[:, 1]).max() > 0.03
