import json

import numpy as np
import pytest

from mkmmd import assets as A
from mkmmd.checks import Context, camera, motion
from mkmmd.project import Project
from mkmmd.solvers import geom


def _shapes(**kinds):
    s = geom.Shapes()
    for kind, items in kinds.items():
        for it in items:
            if kind == "floor":
                s.floor_z = it
            else:
                s.add(kind, ("world", "", ""), kind, **it)
    return s


def _measure(shapes, X, r):
    P = shapes.pack()
    W = geom.world(P, np.eye(3)[None], np.zeros((1, 3)))
    return geom.measure(np.array(X, float), np.full(len(X), r), W, shapes.floor_z, P)


# ---------------------------------------------------------------- penetration geometry
def test_sphere_and_capsule_depths():
    s = _shapes(sphere=[dict(c=[0, 0, 0], R=0.1)], capsule=[dict(a=[1, 0, 0], b=[1, 0, 1], R=0.05)])
    pen, sid = _measure(s, [[0.05, 0, 0], [1.02, 0, 0.5], [3, 3, 3]], 0.01)
    assert pen[0] == pytest.approx(0.06) and sid[0] == 0          # 0.1 + 0.01 - 0.05
    assert pen[1] == pytest.approx(0.04) and sid[1] == 1          # 0.05 + 0.01 - 0.02
    assert pen[2] < 0


def test_box_inside_leaves_by_nearest_face_and_rounded_corner_outside():
    M = np.eye(4)
    s = _shapes(box=[dict(M=M, half=[0.5, 0.2, 0.1], rnd=0.0)])
    pen, _ = _measure(s, [[0.0, 0.0, 0.05]], 0.0)
    assert pen[0] == pytest.approx(0.05)                          # 0.05 below the top face
    s = _shapes(box=[dict(M=M, half=[0.5, 0.2, 0.1], rnd=0.05)])
    corner = np.array([0.5, 0.2, 0.1]) + 0.01                     # just past a rounded corner: outside
    pen, _ = _measure(s, [corner], 0.0)
    assert pen[0] < 0


def test_cylinder_side_and_cap_and_floor():
    M = np.eye(4)
    s = _shapes(cylinder=[dict(M=M, R=0.3, hh=0.05, rnd=0.0)], floor=[0.0])
    pen, sid = _measure(s, [[0.29, 0.0, 0.0], [0.0, 0.0, 0.04], [5.0, 5.0, -0.01]], 0.0)
    assert pen[0] == pytest.approx(0.01) and pen[1] == pytest.approx(0.01)
    assert pen[2] == pytest.approx(0.01) and sid[2] == 1          # the floor is the last label


def test_shapes_ride_on_their_source():
    s = geom.Shapes()
    s.add("sphere", ("object", "", "ball"), "ball", c=[0, 0, 0], R=0.1)
    P = s.pack()
    Rs, ps = np.eye(3)[None], np.array([[2.0, 0.0, 0.0]])
    W = geom.world(P, Rs, ps)
    pen, _ = geom.measure(np.array([[2.0, 0.0, 0.05]]), np.zeros(1), W, None, P)
    assert pen[0] == pytest.approx(0.05)


# ---------------------------------------------------------------- chains + enables
def _rig():
    # one hair chain of two bones hanging from "head" along -z, a head sphere body and an arm capsule body
    bones = {"head": {"head": [0, 0, 1.5]}, "h1": {"head": [0, 0.1, 1.5]}, "h2": {"head": [0, 0.1, 1.3]},
             "arm": {"head": [0.2, 0.1, 1.2]}}
    return {"bones": bones,
            "chains": [{"family": "back_hair", "root": "h1", "anchor": "head", "bones": ["h1", "h2"],
                        "parents": [-1, 0], "ends": [[0, 0.1, 1.3], [0, 0.1, 1.0]], "body_radius": [0.02, 0.02]}],
            "bodies": [{"bone": "head", "group": 0, "no_collide": [], "geom": {"kind": "sphere", "c": [0, 0, 0], "R": 0.12}},
                       {"bone": "arm", "group": 1, "no_collide": [2], "geom": {"kind": "capsule", "a": [0, 0, 0],
                                                                              "b": [0, 0, -0.3], "R": 0.05}},
                       {"bone": "h1", "group": 2, "no_collide": [0, 1]},
                       {"bone": "h2", "group": 2, "no_collide": [0, 1]}]}


def test_chain_points_and_arc():
    ch = geom.Chains(_rig(), ["back_hair"])
    assert ch.bones == ["h1", "h2"] and ch.parent.tolist() == [-1, 0]
    assert ch.arc.tolist() == pytest.approx([0.2, 0.5])
    X, own, frac = ch.points()
    assert X.shape == (6, 3) and own.tolist() == [0, 0, 0, 1, 1, 1]
    assert X[2].tolist() == pytest.approx([0, 0.1, 1.3])


def test_enables_skip_rest_overlap_root_body_and_masks_near_root():
    rig = _rig()
    ch = geom.Chains(rig, ["back_hair"])
    s = geom.Shapes()
    geom.model_shapes(s, rig, "", skip_bones=ch.chain_bones)
    P = s.pack()
    # rest frames of the body sources: bones sit at their heads with identity rotation (armature space)
    heads = {"head": [0, 0, 1.5], "arm": [0.2, 0.1, 1.2]}
    R0 = np.tile(np.eye(3), (len(s.sources), 1, 1))
    p0 = np.array([heads[name] for _, _, name in s.sources], float)
    E = geom.enable_matrix(ch, P, rig, geom.world(P, R0, p0), anchor_free=0.25)
    # sphere: h1 overlaps the head sphere at rest and hangs from it -> off; h2 (0.5 m arc) -> on
    assert E["sphere"][:, 0].tolist() == [False, True]
    # capsule (arm): h1 is near the root and masked against group 1 -> off; h2 is past anchor_free -> on
    assert E["capsule"][:, 0].tolist() == [False, True]
    E2 = geom.enable_matrix(ch, P, rig, geom.world(P, R0, p0), use_masks=True)
    assert E2["capsule"][:, 0].tolist() == [False, False]


def test_quaternion_roundtrip():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(20, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q2 = geom.mat_to_quat(geom.quat_to_mat(q))
    assert np.allclose(np.abs(np.sum(q * q2, 1)), 1.0)


# ---------------------------------------------------------------- contact + tracks
def _project(tmp_path):
    (tmp_path / "mk.toml").write_text('[project]\nfps = 30\nframe0 = 10\nduration = 1\n', encoding="utf-8")
    (tmp_path / "tracks").mkdir()
    (tmp_path / "tracks" / "nib.json").write_text(json.dumps(
        {"frames": [10, 11, 12, 13], "target": [[0, 0, 0], [0, 0, 0], [0, 0, 0], None],
         "writing": [1, 1, 0, 1]}), encoding="utf-8")
    return Project.load(tmp_path)


class FakeData:
    def __init__(self, frames, values):
        self.index = {f: i for i, f in enumerate(frames)}
        self.values = np.array(values, float)

    def expr(self, i, frames):
        return self.values[[self.index[f] for f in frames]]


def test_contact_uses_mask_and_skips_missing_track_values(tmp_path):
    ctx = Context(None, _project(tmp_path))
    m = motion.Contact()
    args = {"a": "obj('Pen').loc", "b": "track:nib.target", "when": "track:nib.writing"}
    from mkmmd.checks import Need
    st = m.needs(args, ctx, Need())
    data = FakeData([10, 11, 12, 13], [[0.001, 0, 0], [0.002, 0, 0], [0.5, 0, 0], [0.9, 0, 0]])
    value, detail = m.compute(args, ctx, data, [10, 11, 12, 13], st)
    assert value == pytest.approx(2.0) and detail["frames"] == 2   # frame 12 is not writing, 13 has no target
    args_z = dict(args, component="x", when=None)
    args_z.pop("when")
    value, detail = m.compute(args_z, ctx, data, [10, 11, 12], st)
    assert value == pytest.approx(500.0) and detail["median"] == pytest.approx(2.0)


# ---------------------------------------------------------------- framing projection
def test_projection_follows_sensor_fit():
    cam = {"world": np.eye(4)[None], "lens": np.array([50.0]), "sensor": np.array([[36.0, 24.0]]),
           "shift": np.zeros((1, 2)), "names": ["Cam"], "sensor_fit": {"Cam": "AUTO"}}
    # a point 1 m in front, 0.36 m right: tan = 0.36 -> x offset = 50/36 * 0.36 = 0.5 of the sensor-fit side
    pt = np.array([[0.36, 0.0, -1.0]])
    uv, depth = camera.project(pt, cam, 0, (1920, 1080))
    assert depth[0] == pytest.approx(1.0) and uv[0, 0] == pytest.approx(1.0)        # right edge of a wide frame
    uv, _ = camera.project(pt, cam, 0, (1080, 1920))
    assert uv[0, 0] == pytest.approx(0.5 + 0.5 * 1920 / 1080)                     # AUTO: sensor on the long side


# the point (0.144 * 36 / 50, 0.086 * 36 / 50, -1) seen through a 50 mm, 36 mm-sensor camera at the origin looking down -Z
# (AUTO fit), with Blender's own bpy_extras.object_utils.world_to_camera_view of a camera with that lens shift, probed in
# Blender 4.2.3 at three sizes: {(w, h): {(shift_x, shift_y): (u, v)}}. The picture moves OPPOSITE to the shift.
BLENDER_SHIFT_UV = {
    (1000, 1000): {(0.0, 0.0): (0.644, 0.586), (0.1, -0.2): (0.544, 0.786), (-0.15, 0.05): (0.794, 0.536),
                   (0.0, -0.259259): (0.644, 0.845259)},
    (1920, 1080): {(0.0, 0.0): (0.644, 0.652889), (0.1, -0.2): (0.544, 1.008444), (-0.15, 0.05): (0.794, 0.564),
                   (0.0, -0.259259): (0.644, 1.113794)},
    (1080, 1920): {(0.0, 0.0): (0.756, 0.586), (0.1, -0.2): (0.578222, 0.786), (-0.15, 0.05): (1.022667, 0.536),
                   (0.0, -0.259259): (0.756, 0.845259)},
}


@pytest.mark.parametrize("size", sorted(BLENDER_SHIFT_UV))
def test_projection_applies_the_lens_shift_as_blender_does(size):
    pt = np.array([[0.144 * 36.0 / 50.0, 0.086 * 36.0 / 50.0, -1.0]])
    for shift, want in BLENDER_SHIFT_UV[size].items():
        cam = {"world": np.eye(4)[None], "lens": np.array([50.0]), "sensor": np.array([[36.0, 24.0]]),
               "shift": np.array([shift]), "names": ["Cam"], "sensor_fit": {"Cam": "AUTO"}}
        uv, depth = camera.project(pt, cam, 0, size)
        assert depth[0] == pytest.approx(1.0)
        assert uv[0] == pytest.approx(want, abs=1e-6), (size, shift)


def test_a_positive_shift_moves_the_picture_left_and_down():
    cam = {"world": np.eye(4)[None], "lens": np.array([50.0]), "sensor": np.array([[36.0, 24.0]]),
           "shift": np.zeros((1, 2)), "names": ["Cam"], "sensor_fit": {"Cam": "AUTO"}}
    pt = np.array([[0.0, 0.0, -1.0]])                                           # on the optical axis
    assert camera.project(pt, cam, 0, (1080, 1920))[0][0] == pytest.approx([0.5, 0.5])
    cam["shift"] = np.array([[0.1, 0.1]])
    u, v = camera.project(pt, cam, 0, (1080, 1920))[0][0]
    assert u < 0.5 and v < 0.5                                                  # left of and below the frame centre
    assert (u, v) == pytest.approx((0.5 - 0.1 * 1920 / 1080, 0.5 - 0.1))        # fractions of the LARGER side


# ---------------------------------------------------------------- registry
def test_registry_credits_flag_unreviewed(tmp_path):
    reg = A.Registry(tmp_path)
    reg.add({"kind": "model", "path": str(tmp_path / "m.pmx"), "slug": "m", "name": "M", "author": "Someone",
             "license": "unreviewed"})
    reg.add({"kind": "audio", "path": str(tmp_path / "s.flac"), "slug": "s", "credit": "Song: Band",
             "license": "personal use"})
    reg.save()
    text, problems = A.Registry(tmp_path).credits(["m", "s"])
    assert "## Models" in text and "- M / Someone" in text and "- Song: Band" in text
    assert problems == ["m: license not reviewed"]
    with pytest.raises(A.AssetError):
        reg.add({"kind": "model", "slug": "m"})
