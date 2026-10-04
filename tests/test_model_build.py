"""mkmmd.model.build: registry, build order, --only planning, context helpers, cross-part checks."""
import numpy as np
import pytest
from PIL import Image

from mkmmd.model import build as BD
from mkmmd.model import part as P
from mkmmd.model import spec as SP


def tri_mesh(name="m", mats=("mat",), weights=None):
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    return P.Mesh(name=name, verts=v, faces=[[0, 1, 2]], uv=np.zeros((3, 2)), mats=list(mats),
                  weights=weights if weights is not None else {"b": np.ones(3)})


@pytest.fixture
def registry():
    saved = dict(BD.REGISTRY)
    BD.REGISTRY.clear()
    yield BD.REGISTRY
    BD.REGISTRY.clear()
    BD.REGISTRY.update(saved)


def make_parts(order):
    @BD.builder("body")
    def _body(ctx):
        order.append("body")
        return P.Part("body", bones=[P.Bone("全ての親", (0, 0, 0))], materials=[P.Material("mat")],
                      meshes=[tri_mesh(weights={"全ての親": np.ones(3)})],
                      bodies=[P.RigidBody("b_head", "全ての親", mode="static")],
                      info={"landmarks": {"head": (0, 0, 1.5), "eye.L": [0.03, -0.05, 1.4]}})

    @BD.builder("head", needs=("body",))
    def _head(ctx):
        order.append("head")
        assert ctx.land["head"].tolist() == [0, 0, 1.5]
        return P.Part("head", bones=[P.Bone("頭", (0, 0, 1.4), parent="全ての親")],
                      materials=[P.Material("skin")], meshes=[tri_mesh("face", ("skin",), {"頭": np.ones(3)})])

    @BD.builder("hair", needs=("body", "head"))
    def _hair(ctx):
        order.append("hair")
        assert ctx.need("head") is ctx.parts["head"]
        return P.Part("hair")

    @BD.builder("outfit")
    def _outfit(ctx):
        order.append("outfit")
        return P.Part("outfit")


def spec_for(parts, **extra):
    d = {"model": {"name": "t", "parts": parts, "out": "unused"}}
    d.update(extra)
    return SP.from_dict(d)


def test_build_order_follows_the_spec(registry, tmp_path):
    order = []
    make_parts(order)
    parts = BD.run(spec_for(["body", "head", "hair", "outfit"]), tex_dir=tmp_path)
    assert order == ["body", "head", "hair", "outfit"]
    assert [p.name for p in parts] == order


def test_only_pulls_in_needs_in_spec_order(registry, tmp_path):
    order = []
    make_parts(order)
    s = spec_for(["body", "head", "hair", "outfit"])
    assert BD.plan(s, ["hair"]) == ["body", "head", "hair"]
    assert BD.plan(s, "head") == ["body", "head"]
    assert BD.plan(s, "outfit") == ["body", "outfit"]               # default needs = body
    assert BD.plan(s, ["body"]) == ["body"]
    with pytest.raises(BD.BuildError, match="nothing"):
        BD.plan(s, ["nothing"])
    BD.run(s, only=["hair"], tex_dir=tmp_path)
    assert order == ["body", "head", "hair"]
    s2 = spec_for(["body", "head", "hair", "outfit"], )
    s2["model"]["needs"] = {"outfit": ["head"], "head": []}           # spec overrides the decorator
    assert BD.plan(s2, ["outfit"]) == ["head", "outfit"]


def test_ctx_rng_is_per_part_and_independent_of_only(registry, tmp_path):
    seen = {}

    @BD.builder("body")
    def _b(ctx):
        seen.setdefault("body", []).append(float(ctx.rng.random()))
        return P.Part("body")

    @BD.builder("hair", needs=())
    def _h(ctx):
        seen.setdefault("hair", []).append(float(ctx.rng.random()))
        return P.Part("hair")
    s = spec_for(["body", "hair"])
    BD.run(s, tex_dir=tmp_path)
    BD.run(s, only=["hair"], tex_dir=tmp_path)
    assert seen["hair"][0] == seen["hair"][1] and seen["body"][0] != seen["hair"][0]
    s["model"]["seed"] = 7
    BD.run(s, only=["hair"], tex_dir=tmp_path)
    assert seen["hair"][2] != seen["hair"][0]


def test_save_png_prefix_formats_and_names(registry, tmp_path):
    ctx = BD.BuildCtx(spec_for(["a"]), tmp_path / "tex")
    ctx.part = "head"
    name = ctx.save_png("skin", np.full((4, 6, 3), 255, np.uint8))
    assert name == "head_skin.png"
    assert ctx.save_png("head_eye.png", np.zeros((2, 2, 4), np.float32)) == "head_eye.png"
    im = Image.open(tmp_path / "tex" / "head_skin.png")
    assert im.mode == "RGBA" and im.size == (6, 4) and im.getpixel((0, 0)) == (255, 255, 255, 255)
    f = ctx.save_png("grad", np.linspace(0, 1, 4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4))
    assert Image.open(tmp_path / "tex" / f).getpixel((3, 3))[0] == 243          # float 60/63 -> 8 bit, rounded
    with pytest.raises(BD.BuildError):
        ctx.save_png("bad name", np.zeros((2, 2, 4)))
    with pytest.raises(BD.BuildError):
        ctx.save_png("x", np.zeros((2, 2, 5)))
    assert ctx.textures == ["head_skin.png", "head_eye.png", f]


def test_wrong_name_wrong_type_and_exceptions(registry, tmp_path):
    @BD.builder("body")
    def _b(ctx):
        return P.Part("other")
    with pytest.raises(BD.BuildError, match="named 'other'"):
        BD.run(spec_for(["body"]), tex_dir=tmp_path)
    BD.REGISTRY.clear()

    @BD.builder("body")
    def _b2(ctx):
        return 5
    with pytest.raises(BD.BuildError, match="expected Part"):
        BD.run(spec_for(["body"]), tex_dir=tmp_path)
    BD.REGISTRY.clear()

    @BD.builder("body")
    def _b3(ctx):
        raise KeyError("boom")
    with pytest.raises(BD.BuildError, match="part 'body' failed: KeyError"):
        BD.run(spec_for(["body"]), tex_dir=tmp_path)
    with pytest.raises(BD.BuildError, match="no builder"):
        BD.get_builder("zzz_missing")


def test_part_check_errors_are_reported_as_build_errors(registry, tmp_path):
    @BD.builder("body")
    def _b(ctx):
        m = tri_mesh()
        m.faces = [[0, 1, 9]]
        return P.Part("body", meshes=[m], materials=[P.Material("mat")])
    with pytest.raises(BD.BuildError, match="bad face"):
        BD.run(spec_for(["body"]), tex_dir=tmp_path)


def test_check_refs_catches_cross_part_mistakes():
    a = P.Part("a", bones=[P.Bone("全ての親", (0, 0, 0)), P.Bone("x", (0, 0, 1), parent="全ての親")],
               materials=[P.Material("m")])
    b = P.Part("b", bones=[P.Bone("y", (0, 0, 2), parent="nobody")], materials=[P.Material("m")],
               meshes=[tri_mesh(weights={"zzz": np.ones(3)}, mats=("nomat",))],
               bodies=[P.RigidBody("r", "ghost")], joints=[P.Joint("j", "r", "q")])
    with pytest.raises(BD.BuildError) as e:
        BD.check_refs([a, b])
    msg = str(e.value)
    for needle in ("unknown parent 'nobody'", "material 'm' defined by both", "unknown bone 'zzz'",
                   "unknown material 'nomat'", "unknown bone 'ghost'", "unknown rigid body 'q'"):
        assert needle in msg
    warns = BD.check_refs([a], strict=False)
    assert isinstance(warns, list)


def test_lint_flags_unweighted_vertices_and_missing_uv():
    m = tri_mesh(weights={"b": np.array([1.0, 0.0, 1.0])})
    m.uv = None
    msgs = BD.lint([P.Part("p", meshes=[m])])
    assert any("1 of 3 vertices have no bone weight" in s for s in msgs)
    assert any("no UV" in s for s in msgs)


def test_find_helpers(registry, tmp_path):
    order = []
    make_parts(order)
    ctx = BD.BuildCtx(spec_for(["body", "head"]), tmp_path)
    BD.run(spec_for(["body", "head"]), ctx=ctx)
    assert ctx.find_body("全ての親").name == "b_head"
    assert ctx.find_body("頭") is None
    assert ctx.find_bone("頭").parent == "全ての親"
    assert set(ctx.bones()) == {"全ての親", "頭"}
    assert ctx.land_point("eye.L").tolist() == [0.03, -0.05, 1.4]
    with pytest.raises(BD.BuildError, match="landmark"):
        ctx.land_point("nope")
