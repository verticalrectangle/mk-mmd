"""Pure maths of the bedroom tech props: lamp arm kinematics, coil spring path, label colour pick, layouts, scribble."""
import math
import random

from mkmmd.blender.library.props import bedroom_tech_maths as M


def _len(v):
    return math.sqrt(sum(c * c for c in v))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


# ----------------------------------------------------------------------------------------------- label colour
def test_label_index_is_stable_and_in_range():
    # md5 based, so the values never change between runs or Python versions
    assert M.label_index("tape1") == 3
    assert M.label_index("tape1") == M.label_index("tape1")
    for name in ("a", "tape", "cassette_pile.001", "x" * 40):
        assert 0 <= M.label_index(name) < len(M.LABEL_SLOTS)


def test_consecutive_numbers_never_repeat_a_colour():
    for prefix in ("tape", "tape_", "cassette.", "mix"):
        picks = [M.label_index(f"{prefix}{i}") for i in range(1, 7)]
        assert sorted(picks) == list(range(len(M.LABEL_SLOTS)))     # all six slots, each once
    # a number must not wrap onto a neighbour inside six tapes, with or without zero padding
    assert len({M.label_index(f"t{i:03d}") for i in range(1, 7)}) == 6


def test_label_slots_are_palette_accents():
    assert set(M.LABEL_SLOTS) == {"love", "gold", "foam", "iris", "rose", "pine"}


def test_seed_of_is_deterministic_and_name_dependent():
    assert M.seed_of("tape1") == M.seed_of("tape1")
    assert M.seed_of("tape1") != M.seed_of("tape2")
    a = M.new_rng("tape1", "label").random()
    assert a == M.new_rng("tape1", "label").random()
    assert a != M.new_rng("tape1", "other").random()


# ----------------------------------------------------------------------------------------------- lamp arms
def test_lamp_pose_matches_the_brief():
    P = M.lamp_joints()
    # the lower arm leans up and back (+y is behind the base, the shade looks toward -y), the upper arm goes forward
    assert P["lower_dir"][2] > 0.9 and P["lower_dir"][1] > 0.0
    assert P["upper_dir"][1] < -0.95
    # the shade looks toward -Y, 35 degrees below the horizon
    out = P["out"]
    assert abs(_len(out) - 1.0) < 1e-9
    assert out[0] == 0.0 and out[1] < 0.0
    assert abs(math.degrees(math.asin(-out[2])) - 35.0) < 1e-9
    # arm lengths are the pivot-to-pivot distances
    assert abs(_len(_sub(P["elbow"], P["pivot"])) - M.LAMP["lower_len"]) < 1e-9
    assert abs(_len(_sub(P["head"], P["elbow"])) - M.LAMP["upper_len"]) < 1e-9
    # the bulb is `neck_back` ahead of the head pivot along the light direction
    d = _sub(P["bulb"], P["head"])
    assert abs(_len(d) - M.LAMP["neck_back"]) < 1e-9
    assert all(abs(a - b * M.LAMP["neck_back"]) < 1e-9 for a, b in zip(d, out))


def test_lamp_height_and_reach_are_about_what_the_brief_asks():
    P = M.lamp_joints()
    assert 0.38 <= P["elbow"][2] <= 0.46                       # the arm's top joint: lamp height ~0.45 m with its knob
    rim_c = tuple(P["bulb"][i] - P["out"][i] * (-0.042) for i in range(3))     # centre of the shade's rim plane
    reach = -rim_c[1] + 0.0675 * math.sqrt(1.0 - P["out"][1] ** 2)           # farthest point of the rim forward of the base
    assert 0.36 <= reach <= 0.44                               # reach ~0.40 m


def test_rot_x_for_turns_z_onto_the_direction():
    for d in ((0.0, 0.2756, 0.9613), (0.0, -0.9986, -0.0523), (0.0, 0.8192, 0.5736)):
        th = M.rot_x_for(d)
        z = (0.0, -math.sin(th), math.cos(th))                 # Rx(th) applied to +Z
        assert all(abs(a - b) < 1e-3 for a, b in zip(z, d))


def test_lamp_joints_override():
    P = M.lamp_joints({"lower_back_deg": 0.0, "upper_up_deg": 0.0, "shade_down_deg": 0.0})
    assert abs(P["elbow"][1] - -M.LAMP["pivot_f"]) < 1e-9      # straight up from the pivot
    assert abs(P["head"][2] - P["elbow"][2]) < 1e-9            # level upper arm
    assert P["out"][2] == 0.0 or abs(P["out"][2]) < 1e-12      # horizontal light


# ----------------------------------------------------------------------------------------------- coil spring
def test_coil_path_runs_between_the_anchors_at_constant_radius():
    a, b = (0.02, -0.01, 0.03), (0.02, 0.01, 0.33)
    pts = M.coil_path(a, b, 0.0058, 0.0105)
    assert pts[0] == a and pts[-1] == b
    ax = _sub(b, a)
    ln = _len(ax)
    u = tuple(c / ln for c in ax)
    radii = []
    for p in pts[1:-1]:
        v = _sub(p, a)
        t = sum(x * y for x, y in zip(v, u))
        radial = tuple(x - t * y for x, y in zip(v, u))
        radii.append(_len(radial))
    assert max(radii) - min(radii) < 1e-9 and abs(radii[0] - 0.0058) < 1e-9
    turns = round((ln - 2 * 0.0035) / 0.0105)
    assert len(pts) == turns * 8 + 3                           # anchor, one point per 1/8 turn (+ the closing one), anchor


def test_coil_path_handles_a_vertical_axis():
    pts = M.coil_path((0, 0, 0), (0, 0, 0.1), 0.005, 0.01)
    assert abs(pts[1][0]) + abs(pts[1][1]) > 0.004             # the helix leaves the axis even when it is along Z


# ----------------------------------------------------------------------------------------------- layouts
def test_spread_is_centred_with_equal_gaps():
    cells = M.spread(5, 0.100, 0.004)
    assert len(cells) == 5
    assert abs(sum(x for x, _ in cells)) < 1e-12               # symmetric about 0
    assert abs(cells[0][1] - (0.100 - 4 * 0.004) / 5) < 1e-12
    gaps = [cells[i + 1][0] - cells[i][0] - cells[i][1] for i in range(4)]
    assert all(abs(g - 0.004) < 1e-12 for g in gaps)
    assert abs(cells[-1][0] + cells[-1][1] / 2 - 0.05) < 1e-12  # the row fills the total width


def test_ticks_span_the_dial_and_mark_every_fifth_and_tenth():
    t = M.ticks(41, 0.074)
    assert len(t) == 41
    assert abs(t[0][0] + 0.037) < 1e-12 and abs(t[-1][0] - 0.037) < 1e-12
    kinds = [k for _, k in t]
    assert kinds[0] == 2 and kinds[10] == 2 and kinds[5] == 1 and kinds[1] == 0
    assert kinds.count(2) == 5


def test_rib_radii_are_even_and_inclusive():
    r = M.rib_radii(0.0125, 0.0400, 6)
    assert r[0] == 0.0125 and abs(r[-1] - 0.0400) < 1e-12
    steps = [b - a for a, b in zip(r, r[1:])]
    assert max(steps) - min(steps) < 1e-12


# ----------------------------------------------------------------------------------------------- scribble
def test_scribble_stays_in_its_box_and_is_deterministic():
    box = (0.075, 0.011)
    a = M.scribble(random.Random(5), *box, lines=2, size=0.0028)
    b = M.scribble(random.Random(5), *box, lines=2, size=0.0028)
    assert a == b and len(a) >= 4
    for pts in a:
        assert len(pts) >= 8
        for x, y in pts:
            assert 0.0 <= x <= box[0] and 0.0 <= y <= box[1]
    c = M.scribble(random.Random(6), *box, lines=2, size=0.0028)
    assert a != c                                               # another name, another handwriting


def test_scribble_is_not_a_straight_line():
    pts = M.scribble(random.Random(1), 0.08, 0.012, lines=1)[0]
    ys = [p[1] for p in pts]
    assert max(ys) - min(ys) > 0.001
