"""form: the blockiness maths (mkmmd.core.form) on synthetic shapes, and the check around it (mkmmd.checks.form) with the
Blender sample faked."""
import os

import numpy as np
import pytest

from mkmmd import bridge
from mkmmd import checks as CH
from mkmmd.core import form as FM
from mkmmd.project import Project


# ---------------------------------------------------------------- shapes (vertices, triangles; normals outward)
def cube_grid(n):
    """The cube [-1, 1]^3 with n x n quads per face."""
    V, T, lin = [], [], np.linspace(-1, 1, n + 1)
    for axis in range(3):
        for sign in (-1, 1):
            u, v = np.meshgrid(lin, lin, indexing="ij")
            P = np.zeros((n + 1, n + 1, 3))
            P[..., axis], P[..., (axis + 1) % 3], P[..., (axis + 2) % 3] = sign, u, v
            idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1) + sum(len(x) for x in V)
            V.append(P.reshape(-1, 3))
            q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
            q = q[:, ::-1] if sign < 0 else q
            T.append(np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]))
    return np.concatenate(V), np.concatenate(T)


def rounded_box(size, r, n=32):
    """A box of full extents `size` whose edges and corners are rounded with radius r (0: sharp)."""
    V, T = cube_grid(n)
    h = np.asarray(size, float) / 2
    P = V * h
    inner = np.clip(P, -(h - r), h - r)
    d = P - inner
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return inner + np.where(norm > 0, d / np.maximum(norm, 1e-12), 0.0) * r, T


def sphere(R=1.0, n=16):
    V, T = cube_grid(n)
    return V / np.linalg.norm(V, axis=1, keepdims=True) * R, T


def cylinder(R, height, seg=48):
    """A closed cylinder about z (sharp rims)."""
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.stack([R * np.cos(a), R * np.sin(a)], 1)
    V = np.concatenate([np.c_[ring, np.full(seg, -height / 2)], np.c_[ring, np.full(seg, height / 2)],
                        [[0, 0, -height / 2], [0, 0, height / 2]]])
    T = []
    for i in range(seg):
        j = (i + 1) % seg
        T += [[i, j, seg + j], [i, seg + j, seg + i], [2 * seg, j, i], [2 * seg + 1, seg + i, seg + j]]
    return V, np.array(T)


def merge(*parts):
    """(V, T, owner) of shapes (V, T) side by side: owner is the shape's index."""
    Vs, Ts, owner, base = [], [], [], 0
    for i, (V, T) in enumerate(parts):
        Vs.append(V)
        Ts.append(T + base)
        owner.append(np.full(len(T), i))
        base += len(V)
    return np.concatenate(Vs), np.concatenate(Ts), np.concatenate(owner)


def moved(V, offset=(0, 0, 0), rot=(0, 0, 0)):
    rx, ry, rz = np.radians(rot)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return V @ (Rz @ Ry @ Rx).T + np.asarray(offset, float)


def score(V, T, owner=None, names=None, **kw):
    return FM.analyse(V, T, owner, names, **kw)


# ---------------------------------------------------------------- the maths
def test_a_cube_is_all_box_and_all_sharp():
    r = score(*rounded_box((1, 1, 1), 0.0, 4))
    assert r["score"] == pytest.approx(1.0)
    assert r["cuboid_share"] == pytest.approx(1.0) and r["flat_share"] == pytest.approx(1.0)
    assert r["hard_edge_share"] == pytest.approx(1.0) and r["flat_hard_share"] == pytest.approx(1.0)
    assert r["parts"] == 1 and r["area"] == pytest.approx(6.0)


def test_smooth_shapes_are_not_blocky():
    for V, T in (sphere(1.0, 24), rounded_box((0.6, 0.6, 1.0), 0.29, 24)):
        r = score(V, T)
        assert r["score"] < 0.05 and r["cuboid_share"] < 0.05


def test_a_cylinder_with_sharp_rims_is_judged_by_its_caps():
    r = score(*cylinder(0.3, 0.8))
    caps = 2 * np.pi * 0.3 ** 2 / (2 * np.pi * 0.3 ** 2 + 2 * np.pi * 0.3 * 0.8)            # 0.27 of the surface
    assert r["cuboid_share"] < 0.05 and caps - 0.01 < r["flat_hard_share"] < caps + 0.06


def test_rounding_an_edge_lowers_the_score_in_steps():
    s = {rad: score(*rounded_box((1, 1, 1), rad))["score"] for rad in (0.0, 0.03, 0.08, 0.1, 0.15, 0.25)}
    assert s[0.0] == pytest.approx(1.0) and s[0.03] > 0.95          # a small bevel on a box is still a box
    assert s[0.1] < s[0.08] < s[0.03]
    assert s[0.15] < 0.25 and s[0.25] < 0.02                        # generous fillets are not


def test_score_is_independent_of_tessellation_and_orientation():
    base = score(*rounded_box((1.0, 0.6, 0.4), 0.05, 24))
    fine = score(*rounded_box((1.0, 0.6, 0.4), 0.05, 40))
    assert fine["score"] == pytest.approx(base["score"], abs=0.06)
    V, T = rounded_box((1.0, 0.6, 0.4), 0.05, 24)
    turned = score(moved(V, (3, -2, 9), (31, -47, 118)), T)
    assert turned["score"] == pytest.approx(base["score"], abs=0.02)
    assert turned["cuboid_share"] == pytest.approx(base["cuboid_share"], abs=0.02)


def test_a_plate_is_judged_by_its_rim_not_as_a_box():
    sharp = score(*rounded_box((1, 1, 0.05), 0.0, 8))
    soft = score(*rounded_box((1, 1, 0.05), 0.015, 32))             # a table top with a rounded rim
    assert sharp["score"] > 0.9 and sharp["flat_hard_share"] > 0.9
    assert soft["score"] < 0.05 and soft["flat_share"] > 0.7 and soft["hard_edge_share"] == 0.0


def test_a_beam_is_a_box_a_plate_is_not():
    n, w, pn, pa = _faces(rounded_box((0.1, 0.1, 1.0), 0.0, 2))
    kappa, share, E = FM.box_weight(n, w, pn, pa)
    assert kappa == pytest.approx(1.0) and share[1] > 0.2 and E is not None
    n, w, pn, pa = _faces(rounded_box((1.0, 1.0, 0.03), 0.0, 2))
    assert FM.box_weight(n, w, pn, pa)[0] == 0.0


def _faces(shape):
    mesh = FM.Mesh(*shape)
    adj = FM.adjacency(mesh)
    label, count = FM.patches(mesh, adj["nbr"])
    pa, pn, _ = FM.patch_table(mesh, label, count)
    return mesh.n, mesh.area, pn, pa


def test_worst_parts_are_named_and_ranked():
    crate = rounded_box((0.5, 0.4, 0.3), 0.0, 4)
    ball = sphere(0.4, 16)
    ball2 = (ball[0] + [1.2, 0, 0], ball[1])
    crate2 = (crate[0] + [0, 1.2, 0], crate[1])
    V, T, owner = merge(ball, crate, ball2, crate2)
    r = score(V, T, owner, ["ball", "crate", "ball2", "crate2"])
    assert {c["object"] for c in r["components"]} == {"crate", "crate2"}
    assert set(r["by_object"]) == {"ball", "crate", "ball2", "crate2"}
    ranked = list(r["by_object"])
    assert set(ranked[:2]) == {"crate", "crate2"} and r["by_object"]["ball"]["score"] < 0.05
    c = r["components"][0]
    assert c["cuboid"] == pytest.approx(1.0) and sorted(c["size"], reverse=True) == pytest.approx([0.5, 0.4, 0.3])
    assert r["patches"] and r["patches"][0]["object"] in ("crate", "crate2") and r["patches"][0]["rim"] == 1.0
    assert 0.2 < r["score"] < 0.5                                    # a third of the surface is boxes


def test_exempt_parts_are_listed_but_weighed_down():
    crate, ball = rounded_box((0.5, 0.4, 0.3), 0.0, 4), sphere(0.5, 16)
    V, T, owner = merge(crate, (ball[0] + [3, 0, 0], ball[1]))
    names = ["wall", "ball"]
    full = score(V, T, owner, names)
    free = score(V, T, owner, names, exempt=["wall"])
    half = score(V, T, owner, names, exempt=["wall"], exempt_weight=0.5)
    assert free["score"] < 0.05 < half["score"] < full["score"]
    assert "wall" in free["exempt"] and "wall" not in free["by_object"] and free["exempt"]["wall"]["score"] > 0.9
    only = score(*merge(crate), names=["wall"], exempt=["wall"])        # nothing else: measured after all
    assert only["score"] > 0.9


def test_loose_parts_and_welding():
    a, b = rounded_box((1, 1, 1), 0.0, 2), rounded_box((1, 1, 1), 0.0, 2)
    V, T, owner = merge(a, (b[0] + [5, 0, 0], b[1]))
    assert score(V, T, owner)["parts"] == 2
    cube_V, cube_T = a
    split = np.concatenate([cube_V, cube_V])                              # the same cube twice, duplicated corners
    only_first = FM.Mesh(split, cube_T)
    assert only_first.nw == len(np.unique(np.round(cube_V, 5), axis=0))
    touching = merge(a, (b[0] + [2, 0, 0], b[1]))                         # share a face plane but not vertices: 2 parts
    assert FM.components(FM.Mesh(*touching)).max() + 1 == 2


def test_adjacency_counts_edges_and_borders():
    cube = FM.Mesh(*rounded_box((1, 1, 1), 0.0, 2))
    adj = FM.adjacency(cube)
    assert adj["border"] == 0.0 and adj["nonmanifold"] == 0
    assert np.isclose(adj["angle"], 90.0).sum() == 12 * 2             # the 12 cube edges, two segments each
    sheet = FM.Mesh(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]), np.array([[0, 1, 2], [0, 2, 3]]))
    adj = FM.adjacency(sheet)
    assert adj["border"] == pytest.approx(4.0) and len(adj["t0"]) == 1 and adj["angle"][0] == pytest.approx(0.0)
    fold = FM.Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 1, 2], [1, 0, 3]]))
    assert FM.adjacency(fold)["angle"][0] == pytest.approx(90.0)


def test_degenerate_triangles_and_empty_meshes():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    T = np.array([[0, 1, 2], [0, 3, 1], [0, 1, 1]])                     # a real triangle, two degenerate ones
    mesh = FM.Mesh(V, T)
    assert len(mesh) == 1 and mesh.index.tolist() == [0]
    empty = score(np.zeros((0, 3)), np.zeros((0, 3), int))
    assert empty["score"] == 0.0 and empty["components"] == [] and empty["triangles"] == 0
    assert score(V, T)["ignored"] == 2


def test_diameter_does_not_depend_on_the_turn():
    V, _ = rounded_box((1, 2, 3), 0.0, 1)
    d = FM.diameter(V)
    assert d == pytest.approx(np.linalg.norm([1, 2, 3]), rel=0.02)
    assert FM.diameter(moved(V, (4, 5, 6), (20, 70, -35))) == pytest.approx(d, rel=0.02)
    assert FM.diameter(np.zeros((0, 3))) == 0.0


def chamfered_cube(c=0.05, h=0.5):
    """A cube of half-size h with its 12 edges cut by 45 degree chamfers c wide (scipy builds the polyhedron)."""
    spatial = pytest.importorskip("scipy.spatial")
    planes = []
    for axis in range(3):
        for s in (-1, 1):
            n = np.zeros(3)
            n[axis] = s
            planes.append(np.r_[n, -h])
    for a, b in ((0, 1), (1, 2), (0, 2)):
        for sa in (-1, 1):
            for sb in (-1, 1):
                n = np.zeros(3)
                n[a], n[b] = sa, sb
                planes.append(np.r_[n, -(2 * h - c)])
    pts = np.unique(np.round(spatial.HalfspaceIntersection(np.array(planes), np.zeros(3)).intersections, 9), axis=0)
    hull = spatial.ConvexHull(pts)
    T = hull.simplices.copy()
    out = np.einsum("ij,ij->i", np.cross(pts[T[:, 1]] - pts[T[:, 0]], pts[T[:, 2]] - pts[T[:, 0]]), hull.equations[:, :3])
    T[out < 0] = T[out < 0][:, ::-1]
    return pts, T


def test_hard_threshold_decides_chamfers():
    # chamfers turn 45 degrees (60 where three of them meet): soft at the default 65, hard at 40; and still a box
    V, T = chamfered_cube()
    soft, hard = score(V, T), score(V, T, hard_deg=40.0)
    assert soft["hard_edge_share"] == 0.0 and hard["hard_edge_share"] == pytest.approx(1.0)
    assert soft["cuboid_share"] > 0.9 and soft["score"] > 0.9 and soft["flat_hard_share"] == 0.0


# ---------------------------------------------------------------- the check
class FakeBlender:
    """Stands in for the Blender `sample` op: writes the .npz the runner reads and returns its JSON reply."""

    def __init__(self, shapes, roots=None, exempt=()):
        self.shapes, self.roots, self.exempt, self.jobs = shapes, roots or {}, exempt, []

    def __call__(self, op, args, **kw):
        assert op == "sample"
        self.jobs.append(args)
        out = {"frames": np.array(args["frames"], int)}
        meta = []
        for k, spec in enumerate(args["meshes"]):
            names = list(self.shapes)
            V, T, owner = merge(*self.shapes.values()) if names else (np.zeros((0, 3)), np.zeros((0, 3), int),
                                                                     np.zeros(0, int))
            out[f"mesh_v_{k}"], out[f"mesh_t_{k}"], out[f"mesh_o_{k}"] = V, T.astype(np.int32), owner.astype(np.int32)
            meta.append({"objects": names, "frame": 7 if spec["frame"] is None else spec["frame"],
                         "roots": {p: self.roots.get(p, np.eye(4)).tolist() for p in (spec["prop"] or [])},
                         "exempt": list(self.exempt)})
        np.savez(args["out"], **out)
        return {"out": args["out"], "frames": len(args["frames"]), "bones": {}, "objects": [], "exprs": [],
                "colliders": [], "meshes": meta}


@pytest.fixture
def blender(monkeypatch, tmp_path):
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    fake = FakeBlender({"car_body": rounded_box((1.0, 0.5, 0.4), 0.0, 3), "wheel": cylinder(0.2, 0.1),
                        "dome": (sphere(0.5, 12)[0] + [0, 3, 0], sphere(0.5, 12)[1])})
    monkeypatch.setattr(bridge, "run", fake)
    return fake


def _run(checks, project=None):
    return CH.run(checks, CH.Context(None, project))


def test_the_check_reports_numbers_and_the_parts_to_fix(blender):
    (res,) = _run([{"name": "car form", "metric": "form", "args": {"prop": "car"}}])
    assert res["metric"] == "form" and res["max"] == 0.25 and res["ok"] is False        # the default limit for hero props
    assert 0.3 < res["value"] < 0.9
    d = res["detail"]
    assert d["worst_parts"][0]["object"] == "car_body" and d["worst_parts"][0]["why"].startswith("a box")
    assert set(d) >= {"cuboid", "flat", "hard_edges", "flat_sharp", "worst_parts", "worst_objects", "sharp_panels", "fix"}
    assert next(iter(d["worst_objects"])) == "car_body"
    assert blender.jobs[0]["meshes"] == [{"objects": [], "prop": ["car"], "exclude": [], "frame": None}]
    assert blender.jobs[0]["frames"] == []                                               # no scene frames needed


def test_limits_come_from_the_check_and_default_to_the_hero_limit(blender):
    checks = [{"name": "a", "metric": "form", "args": {"prop": "car"}, "max": 0.95},
              {"name": "b", "metric": "form", "args": {"prop": "car"}, "max": 0.1},
              {"name": "c", "metric": "form", "args": {"objects": ["dome"], "exclude": ["car_*", "wheel"]}}]
    a, b, c = _run(checks)
    assert a["ok"] and a["max"] == 0.95 and not b["ok"] and b["max"] == 0.1 and "min" not in a
    assert c["max"] == 0.25
    assert len(blender.jobs) == 1 and len(blender.jobs[0]["meshes"]) == 2        # one Blender pass for all three


def test_exempt_by_pattern_by_tag_and_by_the_projects_card(blender, tmp_path):
    base = _run([{"name": "x", "metric": "form", "args": {"prop": "car"}}])[0]
    by_name = _run([{"name": "x", "metric": "form", "args": {"prop": "car", "exempt": ["car_*", "wheel"]}}])[0]
    assert by_name["value"] < base["value"] and by_name["value"] < 0.05 and set(by_name["detail"]["exempt"]) == {"car_body", "wheel"}
    half = _run([{"name": "x", "metric": "form", "args": {"prop": "car", "exempt": "car_body", "exempt_weight": 1}}])[0]
    assert half["value"] == pytest.approx(base["value"], abs=1e-4)
    blender.exempt = ["car_body"]                                                 # the scene tags the object mk_form_exempt
    tagged = _run([{"name": "x", "metric": "form", "args": {"prop": "car"}}])[0]
    assert tagged["value"] < 0.05 and "car_body" in tagged["detail"]["exempt"]
    blender.exempt = []
    (tmp_path / "mk.toml").write_text(
        '[project]\nname = "p"\nfps = 30\nframe0 = 12\nduration = 1\n[[prop]]\nname = "car"\ncard = "library:x"\n'
        'card_extra = { form_exempt = ["car_body"] }\n', encoding="utf-8")
    carded = _run([{"name": "x", "metric": "form", "args": {"prop": "car"}}], Project.load(tmp_path))[0]
    assert carded["value"] < 0.05 and "car_body" in carded["detail"]["exempt"]
    assert blender.jobs[-1]["meshes"][0]["frame"] == 12                          # the project's frame0 by default


def test_the_prop_frame_is_used_for_positions(blender):
    R = np.eye(4)
    R[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]                                # the prop turned 90 degrees
    R[:3, 3] = [10, 20, 0]
    blender.roots = {"car": R}
    plain = _run([{"name": "x", "metric": "form", "args": {"prop": "car"}}])[0]
    assert plain["detail"]["space"] == "prop" and plain["detail"]["at_frame"] == 7
    blender.roots = {}
    world = _run([{"name": "x", "metric": "form", "args": {"prop": ["car", "wheel"]}}])[0]
    assert world["detail"]["space"] == "world"
    assert plain["value"] == pytest.approx(world["value"], abs=1e-3)              # a rigid turn changes no number


def test_the_check_names_its_mistakes(blender):
    (res,) = _run([{"name": "x", "metric": "form", "args": {}}])
    assert res["ok"] is False and "give prop" in res["error"]
    blender.shapes = {}
    (res,) = _run([{"name": "x", "metric": "form", "args": {"prop": "car"}}])
    assert res["ok"] is False and "no visible geometry" in res["error"]


def test_form_is_listed_with_its_default_limit():
    CH.load()
    m = CH.METRICS["form"]
    assert m.default_max == 0.25 and not m.uses_frames and m.sampled and "prop" in m.args


def test_something_scene_sized_gets_a_warning(blender):
    crate, far = rounded_box((0.5, 0.4, 0.3), 0.0, 4), sphere(0.5, 8)
    blender.shapes = {"crate": crate, "far": (far[0] + [100, 0, 0], far[1])}
    (res,) = _run([{"name": "x", "metric": "form", "args": {"prop": "set"}}])
    assert res["detail"]["diameter_m"] == pytest.approx(101, abs=1.0) and "not a prop" in res["detail"]["warning"]
    blender.shapes = {"crate": crate}
    (res,) = _run([{"name": "x", "metric": "form", "args": {"prop": "set"}}])
    assert "warning" not in res["detail"] and res["detail"]["diameter_m"] < 1.0


# ---------------------------------------------------------------- the real Blender op (skipped without Blender)
BUILD_SCENE = '''
import math, bpy, bmesh
from mathutils import Vector

def mesh(name, size, loc, parent, verts_faces=None):
    bm = bmesh.new()
    if verts_faces is None:
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    else:
        vs = [bm.verts.new(v) for v in verts_faces[0]]
        for f in verts_faces[1]:
            bm.faces.new([vs[i] for i in f])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.location, ob.parent = loc, parent
    return ob

root = bpy.data.objects.new("gadget", None)
bpy.context.scene.collection.objects.link(root)
root.location, root.rotation_euler = (5.0, 2.0, 0.0), (0.0, 0.0, math.radians(30))
crate = mesh("gadget_crate", (0.4, 0.3, 0.2), (0.1, 0.2, 0.3), root)
col = mesh("gadget_col", (0.5, 0.5, 0.5), (0, 0, 0), root)
col["mk_collider"] = True
col.hide_render = True
mesh("gadget_hidden", (0.5, 0.5, 0.5), (1, 0, 0), root).hide_render = True
decal = mesh("gadget_decal", (0.3, 0.3, 0.01), (0, 0, 0.6), root)
decal["mk_form_exempt"] = "graphic layer"
octa = ([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)])
ball = mesh("gadget_ball", None, (0.0, -0.8, 0.5), root, octa)
sub = ball.modifiers.new("subsurf", "SUBSURF")
sub.levels, sub.render_levels = 0, 3                 # the viewport shows the octahedron, a render shows a ball
person = bpy.data.objects.new("Person", None)
bpy.context.scene.collection.objects.link(person)
person["mmd_type"] = 2                               # an MMD model's root: what hangs below it is a character
person.parent = root
mesh("Person_mesh", (0.5, 0.5, 1.7), (0, 0, 0), person)
bpy.ops.wm.save_as_mainfile(filepath=OUT)
'''


def have_blender():
    from pathlib import Path

    from mkmmd import config as CFG
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_the_real_sample_op_hands_over_what_a_render_shows(tmp_path):
    import subprocess

    from mkmmd import config as CFG
    cfg = CFG.load()
    out = tmp_path / "gadget.blend"
    subprocess.run([cfg["blender"], "-b", str(bridge.empty_blend(cfg)), "-y", "--python-exit-code", "3", "--python-expr",
                    f"OUT = {str(out)!r}\n" + BUILD_SCENE], check=True, capture_output=True, timeout=300)
    (res,) = CH.run([{"name": "gadget form", "metric": "form", "args": {"prop": "gadget"}}], CH.Context(out, None))
    assert "error" not in res, res
    d = res["detail"]
    assert d["space"] == "prop" and d["objects"] == 3                     # crate, ball, decal: no collider, hidden, character
    assert set(d["exempt"]) == {"gadget_decal"}
    worst = d["worst_parts"][0]
    assert worst["object"] == "gadget_crate" and worst["why"].startswith("a box")
    assert worst["size_m"] == pytest.approx([0.4, 0.3, 0.2], abs=1e-3)
    assert worst["at"] == pytest.approx([0.1, 0.2, 0.3], abs=1e-2)        # the root's turn and place are undone
    assert "gadget_ball" not in d["worst_objects"]                       # Subdivision at render levels: a ball, not an octahedron
    named = CH.run([{"name": "x", "metric": "form", "args": {"objects": ["gadget_ball"]}}], CH.Context(out, None))[0]
    assert named["value"] < 0.1 and named["detail"]["space"] == "world"
    (ex,) = CH.run([{"name": "y", "metric": "form", "args": {"prop": "gadget", "exclude": ["*crate"]}}], CH.Context(out, None))
    assert ex["value"] < 0.1 and ex["detail"]["objects"] == 2
    bad, good = CH.run([{"name": "z", "metric": "form", "args": {"prop": "gadget_cr"}},
                        {"name": "w", "metric": "form", "args": {"prop": "gadget"}}], CH.Context(out, None))
    assert bad["ok"] is False and "no such object" in bad["error"] and "gadget_crate" in bad["error"]      # similar names
    assert good["value"] == pytest.approx(res["value"])                                                  # the other check ran


# ---------------------------------------------------------------- the build-time guard: its policy (pure) and the real build
def test_cards_set_the_limit_and_the_exemptions():
    assert FM.card_limit({}) == FM.HERO_MAX == 0.25 and FM.card_limit({"form_max": 0.7}) == 0.7
    assert FM.is_library("library:chair", {}) and FM.is_library("cards/x.json", {"builder": "library:cassette_player"})
    assert not FM.is_library("pmx:models/mug.pmx", {}) and not FM.is_library("props/x", {"source": "x.blend"})
    names = ["gadget_body", "gadget_label", "gadget_leg_1", "gadget_leg_2", "wall"]
    assert FM.card_exempt(names, ["wall"], {"form_exempt": ["gadget_label", "gadget_leg_*"]}) == \
        ["gadget_label", "gadget_leg_1", "gadget_leg_2", "wall"]
    assert FM.card_exempt(names, [], {"form_exempt": "gadget_body"}) == ["gadget_body"] and FM.card_exempt(names, [], {}) == []


def test_headline_names_the_worst_part_and_fingerprint_ignores_placement():
    V, T, owner = merge(rounded_box((0.5, 0.4, 0.3), 0.0, 4), (sphere(0.5, 8)[0] + [2, 0, 0], sphere(0.5, 8)[1]))
    r = score(V, T, owner, ["crate", "ball"])
    assert FM.headline(r).startswith("crate part 0: a box: flat faces on three axes make up 100%")
    assert FM.headline(score(*sphere(1.0, 8))) is None
    M = np.eye(4)
    M[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    M[:3, 3] = [10.0, -4.0, 2.0]
    world = V @ M[:3, :3].T + M[:3, 3]                                   # the same prop placed and turned in the world
    back = FM.to_frame(world, M)
    assert np.allclose(back, V, atol=1e-9)
    names = ["crate", "ball"]
    key = FM.fingerprint(back + 1e-6, T, owner, names, "src", ["a"])      # float noise below 0.1 mm: the same key
    assert key == FM.fingerprint(V, T, owner, names, "src", ["a"])
    assert key != FM.fingerprint(V + 1e-3, T, owner, names, "src", ["a"])           # a moved vertex: another key
    assert key != FM.fingerprint(V, T, owner, names, "other source", ["a"])
    assert key != FM.fingerprint(V, T, owner, names, "src", ["a", "crate"])


GUARD_PROJECT = """[project]
name = "guard"
fps = 30
frame0 = 1
duration = 1.0
blend = "build/guard.blend"

[look]
palette = "rose-pine-moon"

[[prop]]
name = "mock"
card = "library:car_mockup"
at = [0, 0, 0]

[[prop]]
name = "mock_free"
card = "library:car_mockup"
card_extra = { form_max = 1.0 }
at = [8, 0, 0]

[[prop]]
name = "mock_exempt"
card = "library:car_mockup"
card_extra = { form_exempt = ["*"] }
at = [16, 0, 0]

[[prop]]
name = "mock_limit"
card = "library:car_mockup"
card_extra = { form_max = 0.95 }
at = [24, 0, 0]

[[prop]]
name = "mug"
card = "library:cafe_mug"
at = [32, 0, 0]
"""


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_the_props_stage_warns_about_blocky_library_props(tmp_path, capsys):
    import json

    from mkmmd.cli import main as MAIN
    (tmp_path / "mk.toml").write_text(GUARD_PROJECT, encoding="utf-8")

    def build():
        with pytest.raises(SystemExit) as e:
            MAIN.main(["build", "--until", "props", "--project", str(tmp_path)])
        assert e.value.code == 0
        out = json.loads(capsys.readouterr().out)
        return [ln.split("] ", 1)[1] for ln in out["log"] if "WARNING" in ln]            # without the time stamp
    warnings = build()
    assert len(warnings) == 1 and warnings[0].startswith(
        "WARNING prop 'mock': form 0.90 > 0.25 (mock_tub part 0: flat panels with sharp edges")
    cache = sorted((tmp_path / ".mk" / "cache" / "form").glob("*.json"))
    assert len(cache) == 3                               # mock, mock_limit, mug: free and fully exempt props are not measured
    assert build() == warnings and sorted((tmp_path / ".mk" / "cache" / "form").glob("*.json")) == cache      # from the cache


@pytest.mark.skipif(not have_blender(), reason="Blender not available")
def test_the_library_chair_has_rounded_rims_and_keeps_its_card(tmp_path, capsys):
    import json

    from mkmmd.cli import main as MAIN
    (tmp_path / "mk.toml").write_text(
        '[project]\nname = "chair"\nfps = 30\nframe0 = 1\nduration = 1.0\nblend = "build/chair.blend"\n'
        '[[prop]]\nname = "chair"\ncard = "library:chair"\nat = [0, 0, 0]\n', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        MAIN.main(["build", "--until", "props", "--project", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert e.value.code == 0 and not [ln for ln in out["log"] if "WARNING" in ln]            # the guard has nothing to say
    rep = out["stages"]["props"]["chair"]
    assert rep["uses"] == {"sit": ["seat"], "feet": ["floor"]} and rep["colliders"] == 2 and rep["size"] == [0.43, 0.48, 0.9]
    (res,) = CH.run([{"name": "chair form", "metric": "form", "args": {"prop": "chair"}}],
                    CH.Context(tmp_path / "build" / "chair.blend", None))
    assert res["ok"] and res["value"] < 0.05 and res["detail"]["flat_sharp"] < 0.05       # was 0.55: sharp seat rims, boxy rail
