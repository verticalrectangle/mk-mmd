"""mkmmd.model.partcache: a built part is given back while everything it read is unchanged, and only then."""
import numpy as np
import pytest

from mkmmd.model import build as BD
from mkmmd.model import part as P
from mkmmd.model import partcache as PC
from mkmmd.model import spec as SP


@pytest.fixture
def registry():
    saved = dict(BD.REGISTRY)
    BD.REGISTRY.clear()
    yield BD.REGISTRY
    BD.REGISTRY.clear()
    BD.REGISTRY.update(saved)


def make_parts(runs):
    """A body (reads `loud`, `height`, `mesh` of [body], writes a texture, may log a warning) and a hair on it."""
    @BD.builder("body")
    def _body(ctx):
        runs.append("body")
        if ctx.cfg.get("loud"):
            ctx.log("WARNING the body is loud")
        mesh = ctx.cfg.get("mesh")
        h = float(ctx.cfg.get("height", 1.5)) + (len(open(mesh).read()) * 1e-3 if mesh else 0.0)
        png = ctx.save_png("skin", np.full((2, 2, 3), 200, np.uint8))
        v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        return P.Part("body", bones=[P.Bone("全ての親", (0, 0, 0))], materials=[P.Material("skin", texture=png)],
                      meshes=[P.Mesh("m", verts=v, faces=[[0, 1, 2]], uv=np.zeros((3, 2)), mats=["skin"],
                                     weights={"全ての親": np.ones(3)})],
                      info={"landmarks": {"head": (0, 0, h)}})

    @BD.builder("hair", needs=("body",))
    def _hair(ctx):
        runs.append("hair")
        return P.Part("hair", info={"top": float(ctx.land["head"][2]) + float(ctx.cfg.get("volume", 0.1))})


def test_a_part_is_given_back_while_what_it_read_is_unchanged(registry, tmp_path, monkeypatch):
    """A cached part comes back with its textures and log lines (its warnings) while all it read is unchanged; a key it
    looked up that is newly set, a file a value of it names that changed, new code, or another builder for the part builds
    it again, and every part after it, never one before it; earlier builds stay cached."""
    runs = []
    make_parts(runs)
    cache = PC.PartCache(tmp_path / "cache")
    mesh = tmp_path / "mesh.txt"
    mesh.write_text("abc")

    def build(i, **tables):
        s = SP.from_dict({"model": {"name": "t", "parts": ["body", "hair"], "out": "unused"}, **tables})
        ctx = BD.BuildCtx(s, tmp_path / f"b{i}" / "tex", seed=1)
        runs.clear()
        parts = BD.run(s, ctx=ctx, cache=cache)
        return ctx, parts, list(runs)

    a, pa, ran = build(0, body={"loud": True})
    assert ran == ["body", "hair"] and a.cached == []
    b, pb, ran = build(1, body={"loud": True})
    assert ran == [] and b.cached == ["body", "hair"] and pb[1].info == pa[1].info
    assert b.warnings == a.warnings == ["[body] WARNING the body is loud"]
    assert (tmp_path / "b1" / "tex" / "body_skin.png").read_bytes() == (tmp_path / "b0" / "tex" / "body_skin.png").read_bytes()
    assert build(2, body={"loud": True}, hair={"volume": 0.2})[2] == ["hair"]
    assert build(3, body={"loud": True, "height": 1.6}, hair={"volume": 0.2})[2] == ["body", "hair"]
    assert build(4, body={"loud": True}, hair={"volume": 0.2})[2] == []
    assert build(5, body={"mesh": str(mesh)})[2] == ["body", "hair"]
    assert build(6, body={"mesh": str(mesh)})[2] == []
    mesh.write_text("abcdef")
    assert build(7, body={"mesh": str(mesh)})[2] == ["body", "hair"]
    monkeypatch.setitem(PC._CODE, "d", "another version of the code")
    assert build(8, body={"mesh": str(mesh)})[2] == ["body", "hair"]
    first = BD.REGISTRY["body"].fn

    def stand_in(ctx):                                  # another builder of the part (a project's own, a test's)
        runs.append("stand-in")
        return first(ctx)
    BD.builder("body")(stand_in)
    assert build(9, body={"mesh": str(mesh)})[2] == ["stand-in", "body", "hair"]
