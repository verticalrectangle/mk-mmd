"""bpy-free invariants of the leaf part: its outline (ovate, pointed tip, the bite on the side asked), and on the girl
base's bald head the size, the tip's direction, the stand-off and the curl its keys ask for."""
import numpy as np
import pytest

from mkmmd.model import build as BD
from mkmmd.model import part as PT
from mkmmd.model import spec as SP
from mkmmd.model.parts import leaf as LF


def test_the_outline_is_ovate_with_a_bite_out_of_the_side_asked():
    u = np.linspace(0.0, 1.0, 401)
    c = SP.merge(LF.DEFAULTS, {})
    left, right = LF.half_widths(u, c)
    assert left[0] == right[0] == 0.0 and left[-1] == pytest.approx(0.0, abs=1e-9)
    assert u[np.argmax(right)] == pytest.approx(c["widest"], abs=0.01) and 2 * right.max() == pytest.approx(c["width"])
    at = int(round(c["bite"]["at"] * 400))
    assert right[at] - left[at] == pytest.approx(c["bite"]["depth"], abs=1e-4)      # bitten on its left
    assert np.allclose(left[np.abs(u - c["bite"]["at"]) * c["length"] > c["bite"]["radius"]],
                       right[np.abs(u - c["bite"]["at"]) * c["length"] > c["bite"]["radius"]])
    other = LF.half_widths(u, SP.merge(c, {"bite": {"side": "right"}}))
    assert np.allclose(other[0], right) and np.allclose(other[1], left)


def test_on_a_bald_head_it_lies_where_and_as_its_keys_say(tmp_path):
    spec = SP.load("base:girl", ['model.parts=["body", "head", "leaf"]'])
    ctx = BD.BuildCtx(spec, tmp_path, seed=1)
    BD.run(spec, ctx=ctx)
    part = ctx.parts["leaf"]
    PT.check(part)
    info = part.info
    assert info["length"] == pytest.approx(0.105, abs=0.003) and info["width"] == pytest.approx(0.060, abs=0.002)
    assert info["toward"] == pytest.approx(-35.0, abs=2.0)                       # its tip toward her front right
    assert info["tip"][0] < 0.0 and info["tip"][1] < info["base"][1]
    assert info["off"]["least"] >= 0.002 - 1e-4 and info["off"]["median"] < 0.0045      # 2 mm and more, not floating
    assert info["curl"] == pytest.approx(0.0065, abs=0.0005)
    mesh = part.meshes[0]
    assert list(mesh.weights) == ["頭"] and np.allclose(mesh.weights["頭"], 1.0)
    V = np.asarray(mesh.verts)
    assert V[:, 2].min() > ctx.parts["head"].info["skull_top"][2] - 0.06            # on top of the head
    ctx.part = "leaf"
    ctx.spec["leaf"] = {"lay": {"tilt": 25.0}}                                    # turned toward her front where it lies
    assert LF.build(ctx).info["faces_front"] == pytest.approx(info["faces_front"] + 25.0, abs=3.0)
    ctx.spec["leaf"] = {"lay": {"tilt": 25.0, "over": 0.015}}                     # slid until its tip is over the edge
    slid = LF.build(ctx).info
    assert slid["past_edge"] == pytest.approx(0.015, abs=0.002) and slid["off"]["least"] >= 0.002 - 1e-4
