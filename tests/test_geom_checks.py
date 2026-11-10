import json

import numpy as np
import pytest

from mkmmd import assets as A
from mkmmd.checks import Context, Need, camera, motion, skin
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


# ---------------------------------------------------------------- clearance geometry
def _gaps(shapes, lo, hi):
    W = geom.world(shapes.pack(), np.eye(3)[None], np.zeros((1, 3)))
    return motion.body_gaps(W, np.array(lo, float), np.array(hi, float))


def test_clearance_of_a_sphere_and_a_capsule_from_its_nearest_end():
    s = _shapes(sphere=[dict(c=[0, 0, 0], R=0.1)], capsule=[dict(a=[0, 0, 0], b=[1, 0, 0.2], R=0.05)])
    g = _gaps(s, [[0.5, -1, -1], [-1, -1, 0.5]], [[1, 1, 1], [2, 1, 1]])
    assert g[0, 0] == pytest.approx(0.4) and g[1, 0] == pytest.approx(0.4)        # 0.5 off, less the radius
    assert g[1, 1] == pytest.approx(0.25, abs=1e-9)          # the capsule's raised end, 0.3 under the box


def test_clearance_of_a_box_between_edges_and_through_it():
    turn = np.eye(4)
    turn[:3, :3] = geom.quat_to_mat(np.array([[np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]]))[0]   # 45 deg about z
    s = _shapes(box=[dict(M=turn, half=[0.5, 0.5, 0.5], rnd=0.0)])
    g = _gaps(s, [[1, 1, -1]], [[2, 2, 1]])
    assert g[0, 0] == pytest.approx(np.sqrt(2) - 0.5)       # its top edge to the box's upright edge: no corner is nearer
    stick = np.eye(4)
    stick[:3, :3] = geom.quat_to_mat(np.array([[np.cos(np.pi / 12), 0, 0, np.sin(np.pi / 12)]]))[0]  # 30 deg
    s = _shapes(box=[dict(M=stick, half=[1.0, 0.02, 0.02], rnd=0.01)])
    g = _gaps(s, [[-0.3, -0.3, -0.3], [-0.3, -0.3, 0.0]], [[0.3, 0.3, 0.3], [0.3, 0.3, 0.0]])
    assert g[0, 0] == pytest.approx(-0.01) and g[1, 0] == pytest.approx(-0.01)   # a stick through a box, every
    # corner of each outside the other, and through a flat word


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
    # a scene shape for some families only (a strap the hair lies over, the skirt under it) leaves the other chains alone
    s.add("capsule", ("world", "", ""), "strap", a=[0, 0, 1.0], b=[0.1, 0, 1.0], R=0.02, families=["skirt"])
    s.add("capsule", ("world", "", ""), "strap", a=[0, 0, 1.0], b=[0.1, 0, 1.0], R=0.02, families=["back_hair"])
    P = s.pack()
    p1 = np.vstack([p0, np.zeros((len(s.sources) - len(p0), 3))])
    R1 = np.tile(np.eye(3), (len(s.sources), 1, 1))
    E3 = geom.enable_matrix(ch, P, rig, geom.world(P, R1, p1), anchor_free=0.25)
    assert not E3["capsule"][:, -2].any() and E3["capsule"][:, -1].all()


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
    st = m.needs(args, ctx, Need())
    data = FakeData([10, 11, 12, 13], [[0.001, 0, 0], [0.002, 0, 0], [0.5, 0, 0], [0.9, 0, 0]])
    value, detail = m.compute(args, ctx, data, [10, 11, 12, 13], st)
    assert value == pytest.approx(2.0) and detail["frames"] == 2   # frame 12 is not writing, 13 has no target
    args_z = dict(args, component="x", when=None)
    args_z.pop("when")
    value, detail = m.compute(args_z, ctx, data, [10, 11, 12], st)
    assert value == pytest.approx(500.0) and detail["median"] == pytest.approx(2.0)


class FeetData:
    """Ankle tracks (F, n, 3) as Data.bones and Data.arm_world give them to foot_slide; at rest the ankles stand where
    they do on the first frame."""
    def __init__(self, pos):
        self.pos = np.asarray(pos, float)

    def bones(self, arm, names, frames):
        W = np.tile(np.eye(4), (len(frames), self.pos.shape[1], 1, 1))
        W[..., :3, 3] = self.pos
        rest = np.tile(np.eye(4), (self.pos.shape[1], 1, 1))
        rest[:, :3, 3] = self.pos[0]
        return W, rest, ["ankle.L", "ankle.R"][:self.pos.shape[1]]

    def arm_world(self, arm, frames):
        return np.tile(np.eye(4), (len(frames), 1, 1))


def test_foot_slide_measures_consecutive_frames_only():
    """Two runs of frames with a gap between them: the left foot stands still in each but somewhere else in the second
    (it stepped in the frames not asked for), the right one slides 5 mm a frame. Only the slide counts."""
    frames = list(range(10, 20)) + list(range(40, 50))
    left = [[0.1 if f < 40 else 0.4, 0.0, 0.095] for f in frames]
    right = [[-0.1, 0.005 * f, 0.095] for f in frames]
    value, detail = motion.FootSlide().compute({}, None, FeetData(np.stack([left, right], 1)), frames,
                                               {"arm": "a", "pts": ["ankle.L", "ankle.R"]})
    assert value == pytest.approx(5.0) and detail["at_frame"] in frames
    assert detail["ankle.L"]["max_mm_per_frame"] == 0.0 and detail["ankle.R"]["max_mm_per_frame"] == pytest.approx(5.0)


class FootData:
    """One foot as Data gives it to foot_slide: its ankle bone's world matrices over the frames and its toes' joint's; at
    rest the ankle is 0.09 m over the floor, the toes' joint 0.08 m ahead of it (-y) and 0.03 m up."""
    REST = {"ankle.L": np.array([0.1, 0.0, 0.09]), "toe.L": np.array([0.1, -0.08, 0.03])}

    def __init__(self, ankle, toe):
        self.W = {"ankle.L": np.asarray(ankle, float), "toe.L": np.asarray(toe, float)}

    def has(self, arm, name):
        return name.lstrip("?") in self.W

    def resolve(self, arm, names):
        return [n.lstrip("?") for n in names if n.lstrip("?") in self.W]

    def bones(self, arm, names, frames):
        got = self.resolve(arm, names)
        rest = np.tile(np.eye(4), (len(got), 1, 1))
        rest[:, :3, 3] = [self.REST[n] for n in got]
        return np.stack([self.W[n] for n in got], 1), rest, got

    def arm_world(self, arm, frames):
        return np.tile(np.eye(4), (len(frames), 1, 1))


def test_foot_slide_does_not_take_a_heel_lifting_about_planted_toes_for_a_slide():
    """The heel comes up 10 degrees about the toes' joint and the toes stay put: the ankle travels (up and forward) but
    nothing slides. The same foot pushed 3 mm a frame along the floor does."""
    frames = list(range(20))
    ankle_r, toe_r = FootData.REST["ankle.L"], FootData.REST["toe.L"]
    lifted = np.tile(np.eye(4), (20, 1, 1))
    for k, deg in enumerate(np.linspace(0.0, 10.0, 20)):
        lifted[k, :3, :3] = _rx(deg)
        lifted[k, :3, 3] = toe_r + _rx(deg) @ (ankle_r - toe_r)
    still = np.tile(np.eye(4), (20, 1, 1))
    still[:, :3, 3] = toe_r
    st = {"arm": "a", "pts": ["ankle.L"], "soles": {"ankle.L": "?toe.L"}}
    assert motion.FootSlide().compute({}, None, FootData(lifted, still), frames, st)[0] == pytest.approx(0.0, abs=1e-9)
    head_only = motion.FootSlide().compute({}, None, FootData(lifted, still), frames, {"arm": "a", "pts": ["ankle.L"]})[0]
    assert head_only > 0.4                                       # the ankle's own head goes forward: not what stands
    push = np.outer(np.arange(20), [0.0, -0.003, 0.0])
    flat, toes = np.tile(np.eye(4), (20, 1, 1)), np.tile(np.eye(4), (20, 1, 1))
    flat[:, :3, 3], toes[:, :3, 3] = ankle_r + push, toe_r + push
    assert motion.FootSlide().compute({}, None, FootData(flat, toes), frames, st)[0] == pytest.approx(3.0)


def test_the_ground_under_a_point_is_the_highest_floor_within_reach_not_a_wall_or_a_far_ceiling():
    """A floor, a cove rising 0.5 m over the metre behind it, the wall above the cove and a ceiling at 2 m."""
    V = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [1, 2, 0.5], [-1, 2, 0.5], [1, 2, 3], [-1, 2, 3],
                  [-1, -1, 2], [1, -1, 2], [1, 2, 2], [-1, 2, 2]], float)
    T = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5], [5, 4, 6], [5, 6, 7], [8, 9, 10], [8, 10, 11]]
    pts = np.array([[0.0, 0.0, 0.05], [0.0, 1.5, 0.1], [0.0, 1.5, 0.4], [5.0, 0.0, 0.0]])
    h = motion.surface_height(V, T, pts)
    assert h[0] == pytest.approx(0.0)                             # on the floor: the ceiling overhead is not its ground
    assert h[1] == pytest.approx(0.25) and h[2] == pytest.approx(0.25)     # in the cove (15 cm deep) and over it
    assert h[3] == -np.inf                                        # off the set


# ---------------------------------------------------------------- joint limits: how far the arms turn about themselves
def _rx(deg):
    t = np.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])


def _ry(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])


class RigData:
    """A rig's bones as Data.bones, Data.arm_world and Data.has give them to joint_limits, the armature at the world's
    origin: {bone: (rest head, posed rotations (F, 3, 3), posed heads (F, 3))}, every rest rotation the identity."""
    def __init__(self, bones):
        self.b = bones

    def bones(self, arm, names, frames):
        rows = [self.b[n.lstrip("?")] for n in names]
        W = np.tile(np.eye(4), (len(frames), len(rows), 1, 1))
        rest = np.tile(np.eye(4), (len(rows), 1, 1))
        for j, (head, R, P) in enumerate(rows):
            W[:, j, :3, :3], W[:, j, :3, 3] = R[frames], P[frames]
            rest[j, :3, 3] = head
        return W, rest, names

    def arm_world(self, arm, frames):
        return np.tile(np.eye(4), (len(frames), 1, 1))

    def has(self, arm, name):
        return name.lstrip("?") in self.b


def _t_pose(arms):
    """A standing rig in a T pose (it faces -y, its left is +x), still on every frame but for the arms: `arms` is
    {side: [(the upper arm's rotation, the hand's rotation relative to it)] one per frame}."""
    F = len(next(iter(arms.values())))
    still = lambda head: (np.array(head, float), np.tile(np.eye(3), (F, 1, 1)), np.tile(head, (F, 1)))  # noqa: E731
    bones = {n: still(h) for n, h in (("lower_body", (0, 0, 0.9)), ("upper_body", (0, 0, 0.95)),
                                      ("upper_body2", (0, 0, 1.1)), ("neck", (0, 0, 1.35)), ("head", (0, 0, 1.45)))}
    for s, x in (("L", 1.0), ("R", -1.0)):
        for n, h in (("leg", (0.1 * x, 0, 0.9)), ("knee", (0.1 * x, 0, 0.5)), ("ankle", (0.1 * x, 0, 0.1))):
            bones[f"{n}.{s}"] = still(h)
        sh = np.array([0.15 * x, 0.0, 1.3])
        Ru = np.array([u for u, _ in arms[s]])
        Rh = np.array([u @ h for u, h in arms[s]])
        out = np.array([x, 0.0, 0.0])
        for n, d, R in (("arm", 0.0, Ru), ("elbow", 0.25, Ru), ("wrist", 0.5, Rh), ("middle1", 0.6, Rh)):
            P = sh + d * np.einsum("fij,j->fi", Ru, out) if n != "middle1" else \
                sh + 0.5 * np.einsum("fij,j->fi", Ru, out) + 0.1 * np.einsum("fij,j->fi", Rh, out)
            bones[f"{n}.{s}"] = (sh + d * out, R, P)
    return RigData(bones)


def test_joint_limits_turns_of_the_forearm_and_upper_arm_from_the_rest_mid_prone_roll():
    """Frame 0 the T pose; 1 both hands turned 120 deg palm up (supination: the thumb from forward to up, the same Rx on
    a mirrored pair); 2 both palms turned 100 deg down (pronation); 3 the upper arms turned 120 deg out (external
    rotation) at shoulder height; 4 the same raised 60 deg above it, where an arm turns that far out on its way up; 5
    turned 120 deg in. Supination ends at 100, pronation at -90, the upper arm at +-95 widened by the elevation past
    level."""
    I = np.eye(3)
    poses = [(I, I), (I, _rx(-120)), (I, _rx(100)), (_rx(-120), I), (_ry(-60) @ _rx(-120), I), (_rx(120), I)]
    mirror = np.diag([-1.0, 1.0, 1.0])
    data = _t_pose({"L": poses, "R": [(mirror @ u @ mirror, mirror @ h @ mirror) for u, h in poses]})
    frames = list(range(6))
    st = motion.JointLimits().needs({"armature": "a"}, Context(None), Need())
    value, detail = motion.JointLimits().compute({}, None, data, frames, st)
    for s in ("L", "R"):
        fa, ua = detail[f"forearm_twist.{s}"], detail[f"arm_twist.{s}"]
        assert fa["max"] == pytest.approx(120.0, abs=0.1) and fa["min"] == pytest.approx(-100.0, abs=0.1)
        assert fa["spans"] == [[1, 2, 1, 20.0]] and fa["over_deg"] == pytest.approx(20.0, abs=0.1)
        assert ua["max"] == pytest.approx(120.0, abs=0.1) and ua["min"] == pytest.approx(-120.0, abs=0.1)
        assert ua["spans"] == [[3, 3, 3, 25.0], [5, 5, 5, 25.0]]       # frame 4 is the same turn, raised: inside
    assert value == pytest.approx(25.0, abs=0.1) and detail["worst"].startswith("arm_twist")
    _, detail = motion.JointLimits().compute({"limits": {"forearm_twist": [-110, 130], "arm_twist": [-130, 130]}},
                                              None, data, frames, st)
    assert "over_deg" not in detail["forearm_twist.L"] and "over_deg" not in detail["arm_twist.R"]


# ---------------------------------------------------------------- skin collapse
def _tube(length=0.3, r=0.03, step=0.01, n=16):
    """A tube along +x from the origin: rings of n vertices `step` apart, two triangles a quad."""
    xs = np.arange(0.0, length + 1e-9, step)
    a = 2 * np.pi * np.arange(n) / n
    V = np.array([[x, r * np.cos(t), r * np.sin(t)] for x in xs for t in a])
    T = [[i * n + j, i * n + (j + 1) % n, (i + 1) * n + (j + 1) % n] for i in range(len(xs) - 1) for j in range(n)]
    T += [[i * n + j, (i + 1) * n + (j + 1) % n, (i + 1) * n + j] for i in range(len(xs) - 1) for j in range(n)]
    return V, np.array(T)


class TubeData:
    """A forearm (elbow at the origin, wrist 0.3 m along +x) whose skin is a tube, as Data.skin / bones / arm_world give
    it to skin_collapse: posed (F, n, 3); each vertex's heaviest bone the elbow's up to the middle, the twist bone's
    after it."""
    def __init__(self, V, T, posed):
        self.V, self.T, self.P = V, T, np.asarray(posed)

    def skin(self, i, frames):
        n = len(self.V)
        return {"rest": self.V.astype(np.float32), "posed": self.P[frames].astype(np.float32), "triangles": self.T,
                "region": np.zeros(n, np.int16), "bone": (self.V[:, 0] >= 0.15).astype(np.int32),
                "object": np.zeros(n, np.int16), "regions": ["forearm.L"], "region_bones": {"forearm.L": ["ひじ.L"]},
                "bones": ["ひじ.L", "手捩.L"], "objects": ["sleeve"], "skipped": {}}

    def bones(self, arm, names, frames):
        rest = np.tile(np.eye(4), (2, 1, 1))
        rest[1, 0, 3] = 0.3
        return np.tile(rest, (len(frames), 1, 1, 1)), rest, names

    def arm_world(self, arm, frames):
        return np.tile(np.eye(4), (len(frames), 1, 1))


def test_skin_collapse_finds_the_candy_wrapper_of_a_linear_blend_twist_and_not_a_rigid_or_spread_turn():
    """The tube at rest; turned 170 deg about itself whole (a rigid turn); skinned by linear blend between the elbow and a
    twist bone turned 170 deg, the weights ramping over its middle third (the candy wrapper: the middle ring pulled in to
    cos 85 deg of its radius); each ring turned its share of 170 deg instead (a twist spread along the forearm, the faces
    between the rings necking in only a little)."""
    V, T = _tube()
    w = np.clip((V[:, 0] - 0.1) / 0.1, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    blend = (1.0 - w)[:, None] * V + w[:, None] * (V @ _rx(170).T)
    spread = np.einsum("nij,nj->ni", np.array([_rx(170 * x) for x in w]), V)
    data = TubeData(V, T, [V, V @ _rx(170).T, blend, spread])
    st = skin.SkinCollapse().needs({"armature": "a", "sides": "L", "limbs": "forearm"}, Context(None), Need())
    got = {f: skin.SkinCollapse().compute({}, None, data, [f], st) for f in range(4)}
    assert got[0][0] == pytest.approx(0.0, abs=1e-6) and got[1][0] == pytest.approx(0.0, abs=1e-6)
    value, detail = got[2]
    assert value == pytest.approx(1.0 - np.cos(np.radians(85.0)), abs=0.02)
    assert (detail["kind"], detail["region"], detail["at"], detail["object"]) == ("collapse", "forearm.L", 0.5, "sleeve")
    assert detail["bone"] in ("ひじ.L", "手捩.L") and detail["regions"]["forearm.L"]["loss"] == pytest.approx(value, abs=1e-4)
    assert got[3][0] < 0.06 < skin.SkinCollapse.default_max
    value, detail = skin.SkinCollapse().compute({}, None, data, [0, 1, 2, 3], st)
    assert detail["at_frame"] == 2 and value > skin.SkinCollapse.default_max


def test_a_cut_s_mean_radius_is_the_cross_section_s_size_and_none_where_the_plane_misses():
    V, T = _tube(r=0.05, n=64)
    u = np.array([[1.0, 0.0, 0.0]])
    r, length = skin.cut_radius(V[None], T, np.zeros((1, 3)), u, np.array([0.3]), [0.5, 1.5])
    assert r[0, 0] == pytest.approx(0.05, rel=2e-3) and length[0, 0] == pytest.approx(2 * np.pi * 0.05, rel=2e-3)
    assert np.isnan(r[0, 1]) and length[0, 1] == 0.0



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
    for w in ("regular", "bold"):                     # two weights of one family, credited alike
        reg.add({"kind": "font", "path": str(tmp_path / f"{w}.ttf"), "slug": w, "credit": "Font: Foundry", "license": "OFL"})
    reg.save()
    text, problems = A.Registry(tmp_path).credits(["m", "s", "regular", "bold"])
    assert "## Models" in text and "- M / Someone" in text and "- Song: Band" in text
    assert text.count("- Font: Foundry") == 1
    assert problems == ["m: license not reviewed"]
    with pytest.raises(A.AssetError):
        reg.add({"kind": "model", "slug": "m"})
