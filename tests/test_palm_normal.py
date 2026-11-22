"""The palm's side at rest (mkmmd.blender.build.pose.palm_normal): MMD rests the palms down in the armature's own frame,
so a body lying on its back keeps them down in its frame (decided by the world's z, a body on its back got its palms
turned up). Probed in Blender without rendering (skipped without it)."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mkmmd import bridge
from mkmmd import config as CFG

ROOT = Path(__file__).resolve().parent.parent

PROBE = r'''
import json, math, sys
import bpy
sys.path.insert(0, ROOT)
from mkmmd.blender.build import pose as PO

arm = bpy.data.objects.new("body", bpy.data.armatures.new("body"))
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
for name, head in (("w", (0.6, 0.0, 1.4)), ("i", (0.68, -0.03, 1.41)), ("l", (0.68, 0.03, 1.39))):
    b = arm.data.edit_bones.new(name)           # a left hand at rest, its palm down and a little toward her front
    b.head, b.tail = head, (head[0] + 0.05, head[1], head[2])
bpy.ops.object.mode_set(mode="OBJECT")
smap = {"wrist.L": "w", "index1.L": "i", "little1.L": "l"}
out = {}
for pose, rx in (("standing", 0.0), ("on_her_back", -math.pi / 2)):
    arm.rotation_euler = (rx, 0.0, 0.0)
    bpy.context.view_layer.update()
    out[pose] = {"palm": list(PO.palm_normal(arm, smap, "L")), "rot": [list(r) for r in arm.matrix_world.to_3x3()]}
json.dump(out, open(OUT, "w"))
'''


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_a_body_on_its_back_keeps_its_palms_down_in_its_own_frame(tmp_path):
    out = tmp_path / "probe.json"
    cfg = CFG.load()
    r = subprocess.run([cfg["blender"], "-b", str(bridge.empty_blend(cfg)), "-y", "--python-exit-code", "3", "--python-expr",
                        f"ROOT = {str(ROOT)!r}\nOUT = {str(out)!r}\n" + PROBE], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads(out.read_text())
    stand = np.array(res["standing"]["palm"])
    assert stand == pytest.approx([0.0, -0.3162, -0.9487], abs=1e-3)                # standing: down, as it always was
    lying = np.array(res["on_her_back"]["palm"])
    assert lying == pytest.approx(np.array(res["on_her_back"]["rot"]) @ stand, abs=1e-3)   # on her back: turned with her
