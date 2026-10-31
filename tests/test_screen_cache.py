"""The bake's cache of screen type (mkmmd/blender/build/bake_lit.py `_screen_key`, docs/design.md: Cache): a screen word's
fitted encoding is reused while nothing that makes its geometry changed. Run in Blender on a tiny geometry-nodes word:
a changed keyframe, string or window misses; an unrelated change hits; a word that reads the scene elsewhere (Object
Info) is never cached."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from mkmmd import config as CFG

REPO = Path(__file__).resolve().parents[1]


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:
        return False


SCRIPT = r'''
import json, sys
import bpy
sys.path.insert(0, REPO)
from mkmmd.blender.build import bake_lit as BL

out = sys.argv[sys.argv.index("--") + 1]
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
o = bpy.data.objects.new("word", bpy.data.meshes.new("word"))
bpy.context.scene.collection.objects.link(o)
other = bpy.data.objects.new("other", None)
bpy.context.scene.collection.objects.link(other)
ng = bpy.data.node_groups.new("mk_text_word", "GeometryNodeTree")
ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
stc = ng.nodes.new("GeometryNodeStringToCurves")
stc.inputs["String"].default_value = "ab"
fill, real, tr = (ng.nodes.new(t) for t in ("GeometryNodeFillCurve", "GeometryNodeRealizeInstances",
                                            "GeometryNodeTransform"))
val = ng.nodes.new("ShaderNodeValue")
val.name = "Pop"
comb, gout = ng.nodes.new("ShaderNodeCombineXYZ"), ng.nodes.new("NodeGroupOutput")
ng.links.new(stc.outputs["Curve Instances"], fill.inputs["Curve"])
ng.links.new(fill.outputs["Mesh"], real.inputs["Geometry"])
ng.links.new(real.outputs["Geometry"], tr.inputs["Geometry"])
for i in range(3):
    ng.links.new(val.outputs[0], comb.inputs[i])
ng.links.new(comb.outputs[0], tr.inputs["Scale"])
ng.links.new(tr.outputs["Geometry"], gout.inputs[0])
md = o.modifiers.new("Text", "NODES")
md.node_group = ng
path = 'nodes["Pop"].outputs[0].default_value'
for f, v in ((1, 1.0), (10, 1.3)):
    val.outputs[0].default_value = v
    ng.keyframe_insert(data_path=path, frame=f)
key = lambda k1=20: BL._screen_key(o, {"attr:mk_alpha"}, 0, k1, 1, 30)
res = {"base": key(), "same": key()}
other.location = (1.0, 2.0, 3.0)
bpy.data.materials.new("unrelated").diffuse_color = (1.0, 0.0, 0.0, 1.0)
res["unrelated"] = key()
kp = ng.animation_data.action.fcurves[0].keyframe_points[1]
kp.co[1] = 2.5
res["keyframe"] = key()
kp.co[1] = 1.3
res["restored"] = key()
stc.inputs["String"].default_value = "abc"
res["string"] = key()
stc.inputs["String"].default_value = "ab"
res["window"] = key(21)
info = ng.nodes.new("GeometryNodeObjectInfo")
res["reads_the_scene"] = key()
json.dump(res, open(out, "w"))
'''


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_a_screen_word_is_cached_by_what_makes_it(tmp_path):
    script = tmp_path / "keys.py"
    script.write_text(SCRIPT.replace("REPO", repr(str(REPO)), 1))
    out = tmp_path / "keys.json"
    r = subprocess.run([CFG.load()["blender"], "-b", "--python", str(script), "--", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert out.exists(), r.stdout[-2000:] + r.stderr[-2000:]
    k = json.loads(out.read_text())
    assert k["base"] and k["same"] == k["base"]
    assert k["unrelated"] == k["base"]                  # another object moved, another material: a hit
    assert k["keyframe"] != k["base"]                   # a key's value: a miss
    assert k["restored"] == k["base"]
    assert k["string"] != k["base"]                     # the word's string: a miss
    assert k["window"] != k["base"]                     # the frames it shows on: a miss
    assert k["reads_the_scene"] is None                 # an Object Info node: never cached
