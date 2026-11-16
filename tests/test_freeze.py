"""Freezes (hit-stops; mkmmd.core.freeze, bpy-free): `[[freeze]]` windows in Blender frames and the frame the world is
evaluated at on any frame. The Blender half (mkmmd.blender.freeze): what a held frame leaves on the real clock is probed in
Blender without rendering (skipped without it); the cameras moved by their own keys over a held world are checked by
rendering and looking."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from mkmmd import bridge
from mkmmd import config as CFG
from mkmmd.core import freeze as FZ

ROOT = Path(__file__).resolve().parent.parent


def test_windows_are_frames_from_clip_seconds_sorted():
    assert FZ.normalize([{"from": 3.0, "to": 3.5}, {"from": 1.0, "to": 2.0}], 30, 121) == [[151, 181], [211, 226]]
    assert FZ.normalize([], 30, 121) == []


@pytest.mark.parametrize("specs, frag", [
    ([{"from": 1.0}], "expected {from, to}"),
    ([{"from": 1.0, "to": 2.0, "dur": 1}], "expected {from, to}"),
    ([{"from": 2.0, "to": 2.01}], "at least a frame"),
    ([{"from": 1.0, "to": 2.0}, {"from": 1.5, "to": 3.0}], "overlap")])
def test_bad_freezes_are_refused(specs, frag):
    with pytest.raises(FZ.FreezeError, match=frag):
        FZ.normalize(specs, 30, 121)


def test_the_world_holds_the_first_frame_of_a_window_and_runs_on_at_its_end():
    w = [[151, 181], [211, 226]]
    assert [FZ.source(w, f) for f in (150, 151, 165, 180, 181, 210, 211, 225, 226)] == \
        [150, 151, 151, 151, 181, 210, 211, 211, 226]


PROBE = r'''
import json, sys
import bpy
sys.path.insert(0, ROOT)
from mkmmd.blender import freeze as FRZ

sc = bpy.context.scene

def ramp(owner, path, index=-1):                         # the value is the frame number, frames 1 to 11
    for f in (1, 11):
        target = owner.path_resolve(path.rpartition(".")[0]) if "." in path else owner
        attr = path.rpartition(".")[2]
        if index < 0:
            setattr(target, attr, float(f))
        else:
            getattr(target, attr)[index] = float(f)
        target.keyframe_insert(attr, index=index, frame=f)
    for fc in owner.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

prop = bpy.data.objects.new("prop", None)                # a thing in the world
sc.collection.objects.link(prop)
ramp(prop, "location", 0)
tree = bpy.data.node_groups.new("mk_text_word", "GeometryNodeTree")
size = tree.nodes.new("ShaderNodeValue")
size.name = "Size"
ramp(tree, 'nodes["Size"].outputs[0].default_value')
word = bpy.data.objects.new("word", bpy.data.meshes.new("word"))     # screen type: a word and the tree that sizes it
sc.collection.objects.link(word)
word.modifiers.new("Text", "NODES").node_group = tree
word["mk_screen"], word["mk_show"] = 1, [1, 100]
ramp(word, "location", 0)

def state():
    dg = bpy.context.evaluated_depsgraph_get()
    return {"prop": round(prop.evaluated_get(dg).matrix_world.translation.x, 4),
            "word": round(word.evaluated_get(dg).matrix_world.translation.x, 4),
            "size": round(size.outputs[0].default_value, 4)}

out = {}
FRZ.frame_set(sc, 6, [[3, 8]])
out["held"] = state()
FRZ.frame_set(sc, 9, [[3, 8]])
out["after"] = state()
FRZ.release()
json.dump(out, open(OUT, "w"))
'''


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_a_freeze_holds_the_world_and_screen_type_goes_on(tmp_path):
    out = tmp_path / "probe.json"
    cfg = CFG.load()
    r = subprocess.run([cfg["blender"], "-b", str(bridge.empty_blend(cfg)), "-y", "--python-exit-code", "3", "--python-expr",
                        f"ROOT = {str(ROOT)!r}\nOUT = {str(out)!r}\n" + PROBE], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert res["held"] == {"prop": 3.0, "word": 6.0, "size": 6.0}       # frame 6 of a freeze from 3: the word is at 6
    assert res["after"] == {"prop": 9.0, "word": 9.0, "size": 9.0}      # and everything runs again after it
