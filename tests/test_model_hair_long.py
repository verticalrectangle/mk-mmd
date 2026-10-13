"""Long hair on the girl base: back locks combed behind the ears and leaning behind the shoulders, resting on a garment
built before the hair, the texture's gradient over their whole length; the ahoge; the side bow and its tails. bpy-free."""
import numpy as np
import pytest

from mkmmd.model import build as BD
from mkmmd.model import spec as SP
from mkmmd.model.parts import hair_head as HH
from mkmmd.model.parts import outfit_fit as OF

LONG = ['hair.back.end_above_chin=-0.25', 'hair.back.count=11', 'hair.back.spread=96.0', 'hair.back.lean=0.8',
        'hair.back.behind=0.3', 'hair.back.delta=0.055', 'hair.back.tuck=0.10', 'hair.back.sway=0.02', 'hair.back.turns=3.0',
        'hair.ahoge.enabled=true']


def _build(tmp, overrides):
    spec = SP.load("base:girl", overrides)
    ctx = BD.BuildCtx(spec, tmp, seed=1)
    return ctx, {p.name: p for p in BD.run(spec, ctx=ctx)}


@pytest.fixture(scope="module")
def long_hair(tmp_path_factory):
    """The girl with long hair and a bow, a sweater built before the hair: (ctx, parts)."""
    return _build(tmp_path_factory.mktemp("long"),
                  LONG + ['model.parts=["body", "head", "sweater", "hair", "hair_bow"]'])


def test_long_hair_rests_on_a_garment_built_before_it(long_hair):
    """Every hair vertex below the neck seam keeps BODY_GAP off the sweater as off the skin, on its outside (the
    sweater stands up to 3 cm off the skin: hair kept off the skin alone lies inside the knit)."""
    parts = long_hair[1]
    knit = OF.Skin.from_meshes(parts["sweater"].meshes, "sweater")
    skin = OF.Skin.from_meshes(parts["body"].meshes, "body")
    seam = float(parts["body"].info["neck_top"]["z"])
    V = np.asarray(parts["hair"].meshes[0].verts)
    low = V[V[:, 2] < seam - 0.01]
    d, q, _, _ = knit.closest(low)
    assert np.abs(d).min() > HH.BODY_GAP - 0.0015
    ds, dq = skin.closest(low)[0], skin.closest(q)[0]
    on = (np.abs(d) < ds) & (dq > 0.008)                            # lying on the knit where it stands off the skin
    assert on.sum() > 100                                           # (not at a rolled edge): outside it
    assert (ds[on] > dq[on]).all()


def test_long_hair_chains_start_clear_of_the_colliders(long_hair):
    """Leaning back as they fall, the side-most long locks pass behind the shoulders' colliders: no chain starts inside
    one (the build warns about each that does)."""
    ctx, _ = long_hair
    assert ctx.warnings == []


def test_the_side_most_locks_fall_behind_the_shoulders(long_hair):
    """Combed behind the ears before they fall, the long locks pass the shoulders well behind the shoulder joints
    (not over their tops, where the build used to fold them into flat wings, nor brushing past their backs)."""
    _, parts = long_hair
    L = parts["body"].info["landmarks"]
    V = np.asarray(parts["hair"].meshes[0].verts)
    for side, sg in (("L", 1.0), ("R", -1.0)):
        arm = np.asarray(L[f"arm.{side}"], float)
        at = V[(np.abs(V[:, 2] - arm[2]) < 0.015) & (sg * V[:, 0] > abs(arm[0]) - 0.01)]
        assert len(at) and at[:, 1].min() > arm[1] + 0.035


def test_the_gradient_runs_down_the_whole_length(long_hair):
    """The atlas' root-to-tip gradient (hair_tex.head_v) spans the long locks: their tips are at its end, not most of
    their length clamped there."""
    _, parts = long_hair
    m = parts["hair"].meshes[0]
    V = np.asarray(m.verts)
    corners = np.array([i for f in m.faces for i in f])
    v = np.asarray(m.uv)[:, 1]
    low = V[corners, 2] < float(parts["head"].info["chin"][2]) - 0.05
    assert np.ptp(v[low]) > 0.05                                    # it still changes below the chin


def test_the_ahoge_stands_on_a_chain_above_the_crown(long_hair):
    _, parts = long_hair
    hair = parts["hair"]
    names = [b.name for b in hair.bones]
    assert "アホ毛1" in names and "アホ毛2" in names
    tip = max(b.tail[2] for b in hair.bones if b.name.startswith("アホ毛"))
    assert tip > hair.info["hair_top_z"] + 0.02


def test_no_ahoge_unless_asked(tmp_path):
    _, parts = _build(tmp_path, ['model.parts=["body", "head", "hair"]'])
    assert not [b for b in parts["hair"].bones if b.name.startswith("アホ毛")]


def test_the_bow_tails_hang_down_outside_the_hair(long_hair):
    """Each tail falls most of its length (it does not stick out sideways), outside the hair's outline and off the body,
    on a chain of its own hanging from the head."""
    _, parts = long_hair
    bow = parts["hair_bow"]
    for tag, t in bow.info["tails"].items():
        bones = [b for b in bow.bones if b.name.startswith(f"リボン横{tag}")]
        top, low = bones[0].head[2], bones[-1].tail[2]
        assert top - low > 0.7 * t["length"]
        assert t["clear"] > 0.005
        assert bones[0].parent == "頭"
    assert {rb.mode for rb in bow.bodies} == {"dynamic"}


def test_a_bow_tails_v_cut_stays_beyond_the_row_before_it(long_hair):
    """The V-cut pulls the middle of a tail's end row back towards the knot; with a row closer than that before it the
    ribbon folds over itself there, and Blender gives the fold no normal (the build's verify fails)."""
    _, parts = long_hair
    V = np.asarray(parts["hair_bow"].meshes[0].verts)           # the last tail's rows end the mesh, 6 points each
    end, prev = V[-6:], V[-12:-6]
    ahead = end[[0, 2, 3, 5]].mean(0) - prev[[0, 2, 3, 5]].mean(0)
    for k in (1, 4):                                            # the pulled-back middle, front and back
        assert (end[k] - prev.mean(0)) @ ahead > 0


def test_a_bow_tails_edges_leave_every_vertex_a_normal(long_hair):
    """A tail's edges are narrow flat sides, not knife edges: on the hanging tails no vertex has faces round it looking
    opposite ways (where they do, their normals cancel and Blender gives a long twisting ribbon's edge no normal at
    all: the build's verify fails)."""
    bow = long_hair[1]["hair_bow"]
    m = bow.meshes[0]
    V = np.asarray(m.verts)
    knot = max(b.head[2] for b in bow.bones if b.name.startswith("リボン横") and b.name.endswith("1"))
    total, count = np.zeros_like(V), np.zeros(len(V))
    for f in m.faces:
        P = V[list(f)]
        n = np.cross(P[1] - P[0], P[2] - P[0]) + (np.cross(P[2] - P[0], P[3] - P[0]) if len(f) == 4 else 0.0)
        total[list(f)] += n / np.linalg.norm(n)
        count[list(f)] += 1
    tails = (count > 0) & (V[:, 2] < knot - 0.02)
    assert tails.sum() > 100
    assert (np.linalg.norm(total[tails], axis=1) / count[tails]).min() > 0.25
