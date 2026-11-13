"""The lit bake's material record carries an MMD material's PMX flag "self shadow" (mkmmd/blender/build/bake_lit.py
`_material_doc`, docs/design.md: mkr): a PMX written with the flag off on one material and on on the other, imported
by mmd_tools in Blender, gives `receives_shadow` false and true; a material without mmd_tools' settings receives
shadows. Runs in Blender (no GPU); skipped without it."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from mkmmd import bridge as BR
from mkmmd import config as CFG
from mkmmd.model import pmx_io as PX

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
from mkmmd.blender import runtime as RT
from mkmmd.blender.build import bake_lit as BL

pmx, addon, out = sys.argv[sys.argv.index("--") + 1:]
used = {"groups": set(), "images": set(), "objects": set()}
res = {"plain": BL._material_doc(bpy.data.materials.new("plain"), used)}      # before mmd_tools: no mmd_material
res["mmd_tools"] = RT.enable_mmd(addon)
if res["mmd_tools"]:
    bpy.ops.mmd_tools.import_model(filepath=pmx, types={"MESH", "ARMATURE"}, scale=0.08, log_level="ERROR")
    for mat in bpy.data.materials:
        if mat.mmd_material.name_j in ("shaded", "unshaded"):
            res[mat.mmd_material.name_j] = BL._material_doc(mat, used)
json.dump(res, open(out, "w"))
'''


def _pmx(path):
    """Two triangles, each its own material: "shaded" receives self shadow, "unshaded" does not."""
    vs = [PX.PmxVertex((x, 0.0, z), (0.0, 1.0, 0.0), (0.0, 0.0)) for x, z in ((0, 0), (1, 0), (0, 1), (1, 1))]
    PX.write(PX.PmxModel(name="flags", bones=[PX.PmxBone("全ての親")], vertices=vs, faces=[0, 1, 2, 1, 3, 2],
                         materials=[PX.PmxMaterial("shaded", index_count=3, self_shadow=True),
                                    PX.PmxMaterial("unshaded", index_count=3, self_shadow=False)]), path)


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_the_material_record_says_whether_shadows_fall_on_it(tmp_path):
    pmx = tmp_path / "flags.pmx"
    _pmx(pmx)
    script = tmp_path / "record.py"
    script.write_text(SCRIPT.replace("REPO", repr(str(REPO)), 1))
    out = tmp_path / "record.json"
    cfg = CFG.load()
    # started as mk's jobs start Blender: never with --factory-startup (bridge.empty_blend: it can break mmd_tools)
    r = subprocess.run([cfg["blender"], "-b", str(BR.empty_blend(cfg)), "--python", str(script), "--", str(pmx),
                        cfg.get("mmd_addon", "bl_ext.user_default.mmd_tools"), str(out)],
                       capture_output=True, text=True, timeout=600)
    assert out.exists(), r.stdout[-2000:] + r.stderr[-2000:]
    rec = json.loads(out.read_text())
    assert rec["plain"]["receives_shadow"] is True                    # not an MMD material: shadows fall on it
    assert set(rec["plain"]) == {"settings", "tree", "anim", "keyed", "receives_shadow"}
    if not rec["mmd_tools"]:
        pytest.skip("mmd_tools is not installed in Blender")
    assert rec["shaded"]["receives_shadow"] is True
    assert rec["unshaded"]["receives_shadow"] is False
    # the flag is the PMX's "self shadow" (bit 3), not "self shadow map" (bit 2: it casts), which stays on
    assert rec["unshaded"]["settings"]["mmd_material"]["enabled_self_shadow_map"] is True
