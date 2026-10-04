"""The screen layer in real Blender (mkmmd/blender/styles.py: Looks, docs/design.md: Text, Screen type): a flat plane stands
for a screen text (`mk_screen`, on the plane z = 0, one unit per frame height), and a frame of the cut gets it laid over the
finished picture, at the place the frame heights say, in every shot and both outputs. Skipped without Blender."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from mkmmd import bridge
from mkmmd import config as CFG

ROOT = Path(__file__).resolve().parent.parent

SCENE = r'''
import json, os, sys
import bpy
sys.path.insert(0, ROOT)
import numpy as np
from mkmmd.blender import scene as S
from mkmmd.blender import styles as ST
from mkmmd.core import shotstyle as SS

sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.eevee.taa_render_samples = 1
sc.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.2, 0.2, 0.2, 1)
sc.world = world

def emit_material(name, rgb):
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

def plane(name, size, loc, rgb):
    me = bpy.data.meshes.new(name)
    w, h = size
    me.from_pydata([(-w/2, -h/2, 0), (w/2, -h/2, 0), (w/2, h/2, 0), (-w/2, h/2, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    ob.location = loc
    ob.data.materials.append(emit_material(name, rgb))
    return ob

def camera(name, loc, lens):
    cd = bpy.data.cameras.new(name)
    cd.lens, cd.sensor_fit, cd.sensor_width = lens, "AUTO", 36.0
    ob = bpy.data.objects.new(name, cd)
    sc.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (1.5707963, 0, 0)                  # looks along +y
    return ob

# the world: a green wall at y = 5 that fills every camera's picture
wall = plane("wall", (100, 100), (0, 5, 0), (0.0, 0.5, 0.0))
wall.rotation_euler = (1.5707963, 0, 0)
camera("camA", (0, 0, 0), 28.0)
camera("camB", (3, 0, 1), 85.0)
# two screen type objects: a red box left of centre and a blue one at the top right, in frame heights, made for one output each
r = plane("red@16x9", (0.20, 0.10), (-0.40, 0.10, 0.0), (1.0, 0.0, 0.0))
b = plane("blue@16x9", (0.20, 0.10), (0.50, 0.38, 0.0), (0.0, 0.0, 1.0))
t = plane("red@9x16", (0.20, 0.10), (-0.10, 0.10, 0.0), (1.0, 0.0, 0.0))
for ob, asp, show in ((r, "16x9", [1, 70]), (b, "16x9", [30, 40]), (t, "9x16", [1, 100])):
    ob["mk_screen"], ob["mk_aspect"], ob["mk_show"] = 1, asp, show
    ob.hide_render = ob.hide_viewport = True
fig = plane("fig", (100, 100), (50, 3, 0), (0.3, 0.3, 0.3))              # the figure: the right half of the picture
fig.rotation_euler = (1.5707963, 0, 0)
ko = plane("ko@16x9", (0.60, 0.20), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))      # reversed type across the middle
ko["mk_screen"], ko["mk_knockout"], ko["mk_aspect"], ko["mk_show"] = 1, 1, "16x9", [60, 80]
ko.hide_render = ko.hide_viewport = True
sc["mk_shots"] = json.dumps([
    {"name": "a", "from": 1, "to": 20, "cameras": {"16x9": "camA", "9x16": "camA"}},
    {"name": "b", "from": 20, "to": 50, "cameras": {"16x9": "camB", "9x16": "camB"}},
    {"name": "sil", "from": 50, "to": 60, "cameras": {"16x9": "camA"},
     "styles": {"16x9": {"silhouette": SS.normalize({"style": "silhouette", "colors": {"background": "gold", "subject": "base",
                                                      "accent": "text"}, "hide": ["wall"]}, PALETTE)["silhouette"]}}},
    {"name": "ko", "from": 60, "to": 80, "cameras": {"16x9": "camA"},
     "styles": {"16x9": {"silhouette": SS.normalize({"style": "silhouette", "colors": {"background": "gold", "subject": "base",
                                                      "accent": "text"}, "hide": ["wall"]}, PALETTE)["silhouette"]}}},
])
for name, f0, cam in (("a", 1, "camA"), ("b", 20, "camB"), ("sil", 50, "camA"), ("ko", 60, "camA")):
    sc.timeline_markers.new(name, frame=f0).camera = bpy.data.objects[cam]
sc.camera = bpy.data.objects["camA"]
for f, aspect in ((1, "16x9"), (10, "16x9"), (25, "16x9"), (35, "16x9"), (45, "16x9"), (55, "16x9"), (65, "16x9"), (10, "9x16"), (25, "9x16")):
    sc.frame_set(f)
    w, h = (320, 180) if aspect == "16x9" else (180, 320)
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    S.bind_aspect(aspect, sc)
    sc.frame_set(f)
    lk = ST.Looks(sc)
    kind = lk.prepare(f, aspect)
    path = OUT + f"/f{f}_{aspect}.png"
    lk.render(path)
    px = ST._read_rgba(path)[..., :3]
    lk.close()
    RESULT.append({"frame": f, "aspect": aspect, "kind": kind, "size": [w, h], "camera": sc.camera.name if sc.camera else None,
                   "hidden": [o.name for o in sc.objects if o.hide_render and o.name != "wall"],
                   "reds": np.argwhere((px[..., 0] > 0.8) & (px[..., 1] < 0.2) & (px[..., 2] < 0.2)).tolist()[::7],
                   "blues": np.argwhere((px[..., 2] > 0.8) & (px[..., 0] < 0.2) & (px[..., 1] < 0.2)).tolist()[::7],
                   "mean": px.reshape(-1, 3).mean(0).round(3).tolist(),
                   "at": {k: px[min(int(y * h / 180), h - 1), min(int(x * w / 320), w - 1)].round(3).tolist()
                          for k, (x, y) in {"bg": (20, 20), "fig": (300, 20), "ko_left": (130, 90), "ko_right": (190, 90),
                                            "ko_gap": (140, 40)}.items()}})
# the layer file: a frame with screen type writes `screen/<frame>.png` and keeps the type off the picture
LAYERED = {}
for f in (10, 75):
    sc.frame_set(f)
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 320, 180, 100
    S.bind_aspect("16x9", sc)
    sc.frame_set(f)
    lk = ST.Looks(sc)
    lk.prepare(f, "16x9")
    layer = OUT + f"/frames/screen/{f:05d}.png"
    lk.render(OUT + f"/frames/{f:05d}.png", layer=layer)
    lk.close()
    main = ST._read_rgba(OUT + f"/frames/{f:05d}.png")[..., :3]
    entry = {"layer_file": os.path.exists(layer), "main_reds": int(((main[..., 0] > 0.8) & (main[..., 1] < 0.2)).sum()),
             "main_mean": main.reshape(-1, 3).mean(0).round(3).tolist(),
             "tmp_left": [n for n in os.listdir(os.path.dirname(layer))] if os.path.isdir(os.path.dirname(layer)) else []}
    if entry["layer_file"]:
        lay = ST._read_rgba(layer)
        entry.update(shape=list(lay.shape), alpha_max=float(lay[..., 3].max()),
                     reds=np.argwhere((lay[..., 3] > 0.5) & (lay[..., 0] > 0.8)).tolist()[::7],
                     outside=float(lay[:40, :40, 3].max()))
    LAYERED[f] = entry
json.dump({"frames": RESULT, "layered": LAYERED}, open(OUT + "/result.json", "w"))
'''


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    out = tmp_path_factory.mktemp("screen_layer")
    from mkmmd.core import palette as PAL
    pal = PAL.get("rose-pine-moon", None)
    pre = f"ROOT = {str(ROOT)!r}\nOUT = {str(out)!r}\nRESULT = []\nPALETTE = {json.dumps(dict(pal))}\n"
    r = subprocess.run([CFG.load()["blender"], "-b", str(bridge.empty_blend(CFG.load())), "-y", "--python-exit-code", "3",
                        "--python-expr", pre + SCENE], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads((out / "result.json").read_text())
    return {**{(d["frame"], d["aspect"]): d for d in res["frames"]}, "layered": res["layered"]}


def box(pts):
    ys, xs = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_screen_type_is_laid_over_the_frame_where_the_frame_heights_say_in_every_shot(frames):
    for key in ((10, "16x9"), (25, "16x9")):                          # two shots, two lenses: the same place on the picture
        d = frames[key]
        assert d["kind"] == "screen"
        x0, y0, x1, y1 = box(d["reds"])
        # a 0.2 x 0.1 box at (-0.40, +0.10) frame heights of a 320 x 180 picture (180 px per frame height, centre (160, 90))
        assert (x0 + x1) / 2 == pytest.approx(160 - 0.40 * 180, abs=2.5) and (y0 + y1) / 2 == pytest.approx(90 - 0.10 * 180, abs=2.5)
        assert x1 - x0 == pytest.approx(0.20 * 180, abs=3.5) and y1 - y0 == pytest.approx(0.10 * 180, abs=3.5)
    assert frames[(10, "16x9")]["mean"][1] > 0.2                       # and the shot's own picture (the green wall) is still there


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_it_is_on_screen_only_inside_its_frames_and_for_its_own_output(frames):
    assert frames[(10, "16x9")]["blues"] == [] and frames[(35, "16x9")]["blues"] != []     # `mk_show` = [30, 40)
    x0, y0, x1, y1 = box(frames[(35, "16x9")]["blues"])
    assert (x0 + x1) / 2 == pytest.approx(160 + 0.50 * 180, abs=2.5) and (y0 + y1) / 2 == pytest.approx(90 - 0.38 * 180, abs=2.5)
    assert frames[(45, "16x9")]["blues"] == []
    d = frames[(10, "9x16")]                                          # the 9x16 object is drawn in 9x16 and the 16x9 ones are not
    x0, y0, x1, y1 = box(d["reds"])
    assert (x0 + x1) / 2 == pytest.approx(90 - 0.10 * 320, abs=2.5) and (y0 + y1) / 2 == pytest.approx(160 - 0.10 * 320, abs=2.5)
    assert d["blues"] == []


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_scene_renders_never_show_it_and_a_silhouette_shot_keeps_its_flat_look_with_the_type_over_it(frames):
    d = frames[(55, "16x9")]
    assert d["kind"] == "silhouette"
    assert d["reds"] != []                                           # the red box is on in frames [1, 100): over the silhouette frame
    assert d["mean"][0] > d["mean"][2]                                # the flat gold background (red > blue), not the green wall


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_knockout_type_is_the_figures_ink_on_the_colour_and_the_colour_on_the_figure(frames):
    from mkmmd.core import palette as PAL
    from mkmmd.core import shotstyle as SS
    pal = PAL.get("rose-pine-moon", None)
    gold, base = SS.resolve_colour("gold", pal), SS.resolve_colour("base", pal)
    d = frames[(65, "16x9")]
    assert d["kind"] == "silhouette"
    at = d["at"]
    assert at["bg"] == pytest.approx(list(gold), abs=0.02) and at["fig"] == pytest.approx(list(base), abs=0.02)   # the plain look
    assert at["ko_left"] == pytest.approx(list(base), abs=0.03)           # the type on the colour: the figure's ink
    assert at["ko_right"] == pytest.approx(list(gold), abs=0.03)          # the type across the figure: the colour
    assert at["ko_gap"] == pytest.approx(list(gold), abs=0.02)            # beside the type nothing changes (the strip is hidden)


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_the_layer_file_holds_the_type_and_the_picture_is_left_without_it(frames):
    d = frames["layered"]["10"]
    assert d["layer_file"] and d["shape"] == [180, 320, 4] and d["alpha_max"] == pytest.approx(1.0, abs=0.01)
    x0, y0, x1, y1 = box(d["reds"])
    assert (x0 + x1) / 2 == pytest.approx(160 - 0.40 * 180, abs=2.5) and (y0 + y1) / 2 == pytest.approx(90 - 0.10 * 180, abs=2.5)
    assert d["outside"] == 0.0                                        # transparent where there is no type
    assert d["main_reds"] == 0 and d["main_mean"][1] > 0.2            # the picture is the green wall: the type is not in it
    assert d["tmp_left"] == ["00010.png"]                             # written whole: no temporary file left beside it


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_a_frame_with_no_screen_type_writes_no_layer_and_a_silhouettes_knockout_stays_in_the_picture(frames):
    d = frames["layered"]["75"]
    assert d["layer_file"] is False                                   # only the knockout text is on at 75, and it is the figure's
    assert d["main_mean"][0] > d["main_mean"][2]                      # the flat silhouette picture (gold background)
