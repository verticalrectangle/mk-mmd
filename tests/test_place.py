"""Placement maths without Blender: rectangle gaps, prisms and what blocks what, surfaces from card entries, floor and wall
poses, the rules (explicit, centre, align, facing, near, region, avoid), scatter and the error messages."""
import math

import numpy as np
import pytest

from mkmmd.core import place as P


def rot_z(deg, t=(0.0, 0.0, 0.0)):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    M = np.eye(4)
    M[:2, :2] = [[c, -s], [s, c]]
    M[:3, 3] = t
    return M


def table(w=1.2, d=0.6, z=0.74, owner=None):
    """A table-top surface: a rest plane of an owner placed at `owner` (4x4)."""
    return P.surface_from_entry("top", {"type": "plane", "center": [0, 0, z], "normal": [0, 0, 1], "size": [w, d]},
                                np.eye(4) if owner is None else owner)


def room_floor(w=3.6, d=3.2):
    return P.surface_from_entry("floor", {"center": [0, 0, 0], "normal": [0, 0, 1], "size": [w, d]}, np.eye(4))


def fp(name, w, d, h, ox=0.0, oy=0.0, **kw):
    """A footprint whose bounds are w x d x h around (ox, oy) with the bottom at z = 0."""
    return P.Footprint(name, (ox - w / 2, oy - d / 2, 0.0), (ox + w / 2, oy + d / 2, h), **kw)


def prism(name, cx, cy, hx, hy, z0, z1, yaw=0.0, flat=False):
    return P.Prism(name, P.Rect(cx, cy, hx, hy, yaw), z0, z1, flat)


# ---------------------------------------------------------------- rectangles
def test_gap_apart_touching_overlapping_and_diagonal():
    a = P.Rect(0, 0, 0.5, 0.5)
    assert P.gap(a, P.Rect(1.3, 0, 0.5, 0.5)) == pytest.approx(0.3)            # 0.3 m apart along x
    assert P.gap(a, P.Rect(1.0, 0, 0.5, 0.5)) == pytest.approx(0.0, abs=1e-12)    # touching
    assert P.gap(a, P.Rect(0.8, 0, 0.5, 0.5)) == pytest.approx(-0.2)           # overlap depth 0.2 (shortest move)
    assert P.gap(a, P.Rect(0.7, 0.9, 0.5, 0.5)) == pytest.approx(-0.1)         # overlapping 0.3 in x, 0.1 in y
    assert P.gap(a, P.Rect(1.3, 1.4, 0.5, 0.5)) == pytest.approx(math.hypot(0.3, 0.4))   # corner to corner
    assert P.gap(P.Rect(0, 0, 0.5, 0.5), P.Rect(0, 0, 0.1, 0.1)) == pytest.approx(-0.6)  # one inside the other


def test_gap_turned_rectangles_are_exact():
    a = P.Rect(0, 0, 1.0, 0.1)                                   # a long bar along x
    b = P.Rect(0, 0.5, 1.0, 0.1, math.pi / 2)                    # a bar along y standing on top of it
    assert P.gap(a, b) == pytest.approx(P.gap(b, a))                                     # symmetric
    assert P.gap(a, b) < 0                                       # they cross
    c = P.Rect(2.0, 0.0, 0.3, 0.3, math.radians(45))             # a diamond whose left tip is at x = 2 - 0.424
    assert P.gap(a, c) == pytest.approx(2.0 - 0.3 * math.sqrt(2.0) - 1.0)
    assert P.gap(P.Rect(0, 0, 0.5, 0.5), P.Rect(1.0 + 0.1, 0, 0.5, 0.5, math.radians(10))) > 0


def test_gap_to_point():
    r = P.Rect(1, 1, 0.5, 0.25, math.pi / 2)                     # 0.5 along its y, 0.25 along its x after the turn
    assert P.gap_to_point(r, (1, 1)) == pytest.approx(-0.25)    # centre: depth to the nearest side
    assert P.gap_to_point(r, (1, 2)) == pytest.approx(0.5)      # 0.5 beyond the end (turned: half 0.5 along y)
    assert P.gap_to_point(r, (1.25 + 0.3, 1 + 0.6)) == pytest.approx(math.hypot(0.3, 0.1))


# ---------------------------------------------------------------- prisms and what blocks what
def test_prism_of_a_turned_prop_and_a_tilted_one():
    M = rot_z(90, (2.0, 1.0, 0.5))
    p = P.prism_of("chair", (-0.2, -0.2, 0.0), (0.2, 0.3, 0.9), M)
    assert p.rect.yaw == pytest.approx(math.pi / 2)
    assert p.rect.centre == pytest.approx([2.0 - 0.05, 1.0])      # the local centre (0, 0.05) turned by 90 deg: (-0.05, 0)
    assert (p.rect.hx, p.rect.hy) == pytest.approx((0.2, 0.25))
    assert (p.z0, p.z1) == pytest.approx((0.5, 1.4))
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])      # tipped about x: hull, yaw 0
    q = P.prism_of("tilted", (-0.5, -0.5, 0.0), (0.5, 0.5, 1.0), T)
    assert q.rect.yaw == 0.0 and q.z1 > 1.0 and q.rect.hy > 0.5


def test_blocking_rules_heights_flat_and_avoid():
    new = prism("lamp", 0, 0, 0.1, 0.1, 0.74, 1.2)
    desk = prism("desk", 0, 0, 0.6, 0.3, 0.0, 0.74)
    assert P.conflict(new, desk, 0.02) is None                             # standing on the desk top: heights only touch
    tape = prism("tape", 0.05, 0, 0.05, 0.03, 0.74, 0.7525)
    assert P.conflict(new, tape, 0.02) == pytest.approx(P.gap(new.rect, tape.rect))
    assert P.conflict(new, tape, 0.02) < 0                                 # overlapping at the same height
    far = prism("far", 0.25, 0, 0.1, 0.1, 0.74, 1.0)
    assert P.conflict(new, far, 0.02) is None                              # 0.05 m apart is clear enough
    assert P.conflict(new, far, 0.08) == pytest.approx(0.05)               # closer than a 80 mm clear
    rug = prism("rug", 0, 0, 1, 1, 0.0, 0.012, flat=True)
    chair = prism("chair", 0, 0, 0.2, 0.2, 0.0, 0.9)
    assert P.conflict(chair, rug, 0.02) is None                            # a flat rug never blocks
    assert P.conflict(chair, rug, 0.02, forced=True) < 0                   # unless it is listed in `avoid`
    assert P.conflict(rug, chair, 0.02) is None                            # and it is not blocked either: it lies under things
    above = prism("shelf", 0, 0, 0.2, 0.2, 1.5, 1.7)
    assert P.conflict(chair, above, 0.02) is None                          # passes under a shelf


# ---------------------------------------------------------------- surfaces
def test_surface_from_plane_edge_disc_and_wall():
    s = table(owner=rot_z(180, (1.0, 2.0, 0.0)))
    assert s.kind == "floor" and s.origin == pytest.approx((1.0, 2.0, 0.74)) and s.half == pytest.approx((0.6, 0.3))
    assert s.u == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12) and s.v == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)
    assert s.angle == pytest.approx(math.pi)
    disc = P.surface_from_entry("top", {"type": "plane", "center": [0, 0, 0.74], "normal": [0, 0, 1], "radius": 0.32},
                                np.eye(4))
    assert disc.shape == "disc" and disc.half[0] == pytest.approx(0.32)
    edge = P.surface_from_entry("rim", {"type": "edge", "a": [-0.14, 0.28, 0.74], "b": [0.14, 0.28, 0.74],
                                        "normal": [0, 0, 1]}, np.eye(4))
    assert edge.kind == "edge" and edge.half == pytest.approx((0.14, 0.0)) and edge.origin == pytest.approx((0, 0.28, 0.74))
    wall = P.surface_from_entry("wall_back", {"center": [0, 1.6, 1.275], "normal": [0, -1, 0], "up": [0, 0, 1],
                                              "size": [3.6, 2.55]}, np.eye(4))
    assert wall.kind == "wall" and wall.u == pytest.approx((1, 0, 0)) and wall.v == (0, 0, 1)
    left = P.surface_from_entry("wall_left", {"center": [-1.8, 0, 1.275], "normal": [1, 0, 0], "size": [3.2, 2.55]},
                                np.eye(4))
    assert left.u == pytest.approx((0, 1, 0))                 # to the right as seen from the room (looking toward -x)


def test_surface_errors_say_what_is_wrong():
    ent = {"center": [0, 0, 0], "normal": [0, 0, 1]}
    with pytest.raises(P.PlaceError, match="neither a `size` nor a `radius`"):
        P.surface_from_entry("top", ent, np.eye(4))
    lo_hi = ((-0.3, -0.2, 0.0), (0.3, 0.2, 0.5))
    s = P.surface_from_entry("top", ent, np.eye(4), bounds=lo_hi)
    assert s.half == pytest.approx((0.3, 0.2))                 # the owner's bounds stand in
    with pytest.raises(P.PlaceError, match="faces down"):
        P.surface_from_entry("ceiling", {"center": [0, 0, 2], "normal": [0, 0, -1], "size": [1, 1]}, np.eye(4))
    with pytest.raises(P.PlaceError, match="tilted"):
        P.surface_from_entry("ramp", {"center": [0, 0, 0], "normal": [0, 0.5, 0.866], "size": [1, 1]}, np.eye(4))
    with pytest.raises(P.PlaceError, match="tilted"):
        P.surface_from_entry("sloped wall", {"center": [0, 0, 1], "normal": [0, -0.9, 0.3], "size": [1, 1]}, np.eye(4))
    with pytest.raises(P.PlaceError, match="needs a `size`"):
        P.surface_from_entry("wall", {"center": [0, 0, 1], "normal": [0, -1, 0]}, np.eye(4))


def test_bounds_from_a_cards_size():
    assert P.bounds_from_size((0.4, 0.6, 0.9)) == ((-0.2, -0.3, 0.0), (0.2, 0.3, 0.9))
    assert P.bounds_from_size((0.4, 0.6, 0.9), "center") == ((-0.2, -0.3, -0.45), (0.2, 0.3, 0.45))
    assert P.bounds_from_size((0.3, 0.01, 0.4), "wall_center") == ((-0.15, -0.01, -0.2), (0.15, 0.0, 0.2))


# ---------------------------------------------------------------- poses
def test_floor_pose_puts_the_bounds_on_the_surface():
    s = table()
    chair = P.Footprint("chair", (-0.2, -0.2, 0.0), (0.2, 0.3, 0.9))               # origin not at the footprint centre
    pose = P.pose_on(s, chair, 0.1, 0.05, 0.0)
    assert pose.origin == pytest.approx([0.1, 0.05 - 0.05, 0.74])                  # footprint centre (0, 0.05) from origin
    assert (pose.prism.z0, pose.prism.z1) == pytest.approx((0.74, 1.64))
    pose = P.pose_on(s, chair, 0.0, 0.0, math.pi)                                  # turned half way round
    assert pose.origin[:2] == pytest.approx([0.0, 0.05])                           # the centre offset turns with it
    below = P.Footprint("sunk", (-0.1, -0.1, -0.02), (0.1, 0.1, 0.3))              # bounds below the origin
    assert P.pose_on(s, below, 0, 0, 0).origin[2] == pytest.approx(0.76)


def test_wall_pose_hangs_with_the_back_or_the_origin_on_the_plane():
    wall = P.surface_from_entry("wall_back", {"center": [0, 1.6, 1.275], "normal": [0, -1, 0], "size": [3.6, 2.55]},
                                np.eye(4))
    poster = P.Footprint("poster", (-0.25, -0.008, -0.35), (0.25, -0.002, 0.35), wall_origin=True)
    pose = P.pose_on(wall, poster, 0.4, 0.3, 0.0)
    assert pose.yaw == pytest.approx(0.0)
    assert pose.origin == pytest.approx([0.4, 1.6, 1.275 + 0.3])                   # origin on the plane
    assert pose.corners[:, 0].min() == pytest.approx(0.15) and pose.corners[:, 1].max() == pytest.approx(0.65)
    shelf = P.Footprint("shelf", (-0.4, -0.1, 0.0), (0.4, 0.1, 0.05))              # origin at the bottom centre
    pose = P.pose_on(wall, shelf, 0.0, 0.0, 0.0)
    assert pose.prism.rect.cy + pose.prism.rect.hy == pytest.approx(1.6)           # the back of its bounds on the wall
    left = P.surface_from_entry("wall_left", {"center": [-1.8, 0, 1.275], "normal": [1, 0, 0], "size": [3.2, 2.55]},
                                np.eye(4))
    pose = P.pose_on(left, poster, 0.5, -0.2, 0.0)
    assert pose.yaw == pytest.approx(math.pi / 2)                                  # -Y turned to face +X
    assert pose.origin == pytest.approx([-1.8, 0.5, 1.275 - 0.2])                  # u runs along +y on this wall
    side = P.Footprint("sidefront", (-0.2, -0.05, -0.2), (0.2, 0.05, 0.2), front="+X", wall_origin=True)
    assert P.pose_on(left, side, 0, 0, 0).yaw == pytest.approx(0.0)                # +X already faces out of this wall
    assert P.pose_on(wall, side, 0, 0, 0).yaw == pytest.approx(-math.pi / 2)       # on the back wall it turns to face -Y


# ---------------------------------------------------------------- rules on a table top
PLAYER = fp("player", 0.34, 0.115, 0.16)


def test_explicit_and_centre_placements():
    top = table()
    res = P.place(top, PLAYER, [], P.Rule(at=(0.2, 0.05)))
    assert res.pose.uv == (0.2, 0.05) and res.pose.origin == pytest.approx([0.2, 0.05, 0.74])
    assert res.clearance["edge"] == pytest.approx(0.3 - 0.05 - 0.0575)             # 0.1925 m from the back edge
    assert res.clearance["prop"] is None
    c = P.place(top, PLAYER, [], P.Rule(at="center"))
    assert c.pose.uv == (0.0, 0.0)
    rep = res.report()
    assert rep["on"] == "top" and rep["position"] == [0.2, 0.05, 0.74] and rep["yaw_deg"] == 0.0
    assert rep["clearance_mm"] == {"edge": 192.5, "prop": None}


def test_a_footprint_off_the_surface_is_an_error_with_numbers():
    with pytest.raises(P.PlaceError) as e:
        P.place(table(), PLAYER, [], P.Rule(at=(0.5, 0.0)))
    msg = str(e.value)
    assert "player" in msg and "right edge" in msg and "'top'" in msg
    assert "90 mm too close" in msg                                                # 0.6 - 0.02 - (0.5 + 0.17) = -0.09
    with pytest.raises(P.PlaceError, match="front edge"):
        P.place(table(), PLAYER, [], P.Rule(at=(0.0, -0.25)))
    with pytest.raises(P.PlaceError, match="`at` must be"):
        P.place(table(), PLAYER, [], P.Rule(at="somewhere"))


def test_align_pushes_against_the_edges_with_the_clearance():
    top = table()
    r = P.place(top, PLAYER, [], P.Rule(at=(0.2, 0.0), align=("front",), clear=0.02))
    assert r.pose.uv[0] == pytest.approx(0.2) and r.pose.uv[1] == pytest.approx(-0.3 + 0.02 + 0.0575)
    assert r.clearance["edge"] == pytest.approx(0.02)
    r = P.place(top, PLAYER, [], P.Rule(align=("back",), clear=0.0))
    assert r.pose.uv[1] == pytest.approx(0.3 - 0.0575)
    r = P.place(top, PLAYER, [], P.Rule(at=(0.0, -0.1), align=("left", "back"), clear=0.01))        # a corner
    assert r.pose.uv == pytest.approx((-0.6 + 0.01 + 0.17, 0.3 - 0.01 - 0.0575))
    r = P.place(top, PLAYER, [], P.Rule(align=("right",)))
    assert r.pose.uv[0] == pytest.approx(0.6 - 0.02 - 0.17)
    with pytest.raises(P.PlaceError, match="both claim the same axis"):
        P.place(top, PLAYER, [], P.Rule(align=("left", "right")))
    with pytest.raises(P.PlaceError, match="not one of"):
        P.place(top, PLAYER, [], P.Rule(align=("middle",)))
    turned = P.place(top, PLAYER, [], P.Rule(facing=("yaw", 90), align=("front",)))
    assert turned.pose.uv[1] == pytest.approx(-0.3 + 0.02 + 0.17)                  # turned 90: its long side is now in y


def test_align_on_a_round_surface():
    top = P.surface_from_entry("top", {"type": "plane", "center": [0, 0, 0.74], "normal": [0, 0, 1], "radius": 0.32},
                               np.eye(4))
    cup = fp("cup", 0.08, 0.08, 0.1)
    r = P.place(top, cup, [], P.Rule(align=("front",), clear=0.02))
    assert r.pose.uv[0] == pytest.approx(0.0)
    assert math.hypot(0.04, abs(r.pose.uv[1]) + 0.04) == pytest.approx(0.32 - 0.02, abs=1e-6)       # corner on the circle
    with pytest.raises(P.PlaceError, match="one side only"):
        P.place(top, cup, [], P.Rule(align=("front", "left")))
    with pytest.raises(P.PlaceError, match="round"):
        P.place(top, cup, [], P.Rule(at=(0.3, 0.0)))


def test_facing_a_point_a_yaw_and_the_surface_frame():
    top = table()
    r = P.place(top, PLAYER, [], P.Rule(at=(0.0, 0.0), facing=("point", (0.0, -2.0))))
    assert r.pose.yaw == pytest.approx(0.0)                                         # front is -Y: already facing it
    r = P.place(top, PLAYER, [], P.Rule(at=(0.0, 0.0), facing=("point", (0.0, 2.0))))
    assert abs(r.pose.yaw) == pytest.approx(math.pi)
    r = P.place(top, PLAYER, [], P.Rule(at=(0.0, 0.0), facing=("point", (2.0, 0.0))))
    assert r.pose.yaw == pytest.approx(math.pi / 2)                                 # -Y turned to +X
    side = P.Footprint("sofa", (-0.4, -0.1, 0.0), (0.4, 0.1, 0.1), front="+X")
    r = P.place(room_floor(), side, [], P.Rule(at=(0.0, 0.0), facing=("point", (0.0, -1.0))))
    assert r.pose.yaw == pytest.approx(-math.pi / 2)                                # +X turned to face -Y
    with pytest.raises(P.PlaceError, match="cannot face"):
        P.place(top, PLAYER, [], P.Rule(at=(0.1, 0.0), facing=("point", (0.1, 0.0))))
    # yaws are relative to the surface: the same rule on a desk turned 90 deg gives a world yaw 90 deg more
    turned = table(owner=rot_z(90))
    r = P.place(turned, P.Footprint("p", (-0.05, -0.05, 0), (0.05, 0.05, 0.05)), [], P.Rule(yaw=20.0))
    assert r.yaw_deg == pytest.approx(110.0)
    r = P.place(turned, P.Footprint("p", (-0.05, -0.05, 0), (0.05, 0.05, 0.05)), [], P.Rule(facing=("yaw", -30)))
    assert r.yaw_deg == pytest.approx(60.0)


def test_other_props_block_unless_they_are_underneath_or_flat():
    top = table()
    lamp = prism("lamp", -0.2, 0.0, 0.08, 0.08, 0.74, 1.2)
    with pytest.raises(P.PlaceError, match="overlaps 'lamp'"):
        P.place(top, PLAYER, [lamp], P.Rule(at=(-0.1, 0.0)))
    r = P.place(top, PLAYER, [lamp], P.Rule(at=(0.2, 0.0)))                          # its left edge at 0.03, the lamp's at -0.12
    assert r.clearance["prop"][0] == "lamp" and r.clearance["prop"][1] == pytest.approx(0.15)
    with pytest.raises(P.PlaceError, match="only 25 mm from 'lamp'"):
        P.place(top, PLAYER, [lamp], P.Rule(at=(0.075, 0.0), clear=0.1))
    desk = prism("desk", 0, 0, 0.6, 0.3, 0.0, 0.74)
    assert P.place(top, PLAYER, [desk], P.Rule(at=(0.2, 0.0))).clearance["prop"] is None   # it stands on the desk
    rug = prism("rug", 0, 0, 1, 1, 0.0, 0.012, flat=True)
    floor = room_floor()
    chair = fp("chair", 0.45, 0.5, 0.9)
    assert P.place(floor, chair, [rug], P.Rule(at=(0, 0))).pose.origin[2] == 0.0
    with pytest.raises(P.PlaceError, match="overlaps 'rug'"):
        P.place(floor, chair, [rug], P.Rule(at=(0, 0), avoid=("rug",)))
    new_rug = P.Footprint("rug", (-0.9, -0.6, 0.0), (0.9, 0.6, 0.012), flat=True)       # a rug placed after the chair
    stand = prism("chair", 0, 0, 0.2, 0.2, 0.0, 0.9)
    assert P.place(floor, new_rug, [stand], P.Rule(at=(0, 0))).pose.uv == (0.0, 0.0)


def test_avoid_keeps_clear_whatever_the_heights():
    floor = room_floor()
    poster_clear = prism("pic", 0, 0, 0.2, 0.2, 1.5, 1.7)
    chair = fp("chair", 0.45, 0.5, 0.9)
    assert P.place(floor, chair, [poster_clear], P.Rule(at=(0, 0))).pose.uv == (0.0, 0.0)       # passes under
    with pytest.raises(P.PlaceError, match="overlaps 'pic'"):
        P.place(floor, chair, [poster_clear], P.Rule(at=(0, 0), avoid=("pic",)))


# ---------------------------------------------------------------- near
def test_near_a_prop_at_an_exact_gap_and_bearing():
    floor = room_floor()
    desk = prism("desk", 0.4, 1.28, 0.6, 0.3, 0.0, 0.74)
    chair = fp("chair", 0.45, 0.5, 0.9, oy=0.05)
    rule = P.Rule(at=P.Near(desk.rect, 0.05, 0.0, "desk", 0.0), facing=("point", (0.4, 1.28)))
    r = P.place(floor, chair, [desk], rule)
    assert P.gap(r.pose.prism.rect, desk.rect) == pytest.approx(0.05, abs=1e-6)
    assert r.pose.prism.rect.cx == pytest.approx(0.4, abs=1e-6)                      # in front of the desk (-Y side)
    assert r.pose.prism.rect.cy < desk.rect.cy and abs(r.pose.yaw) == pytest.approx(math.pi, abs=1e-6)  # facing +Y
    tucked = P.place(floor, chair, [desk], P.Rule(at=P.Near(desk.rect, -0.05, 0.0, "desk"), facing=("point", (0.4, 1.28))))
    assert P.gap(tucked.pose.prism.rect, desk.rect) == pytest.approx(-0.05, abs=1e-6)  # pulled in under the desk edge
    assert tucked.clearance["prop"] is None                                           # the target is exempt
    left = P.place(floor, chair, [desk], P.Rule(at=P.Near(desk.rect, 0.1, 90.0, "desk")))
    assert left.pose.prism.rect.cx > desk.rect.cx + 0.6                               # +90 deg: to the target's +X side


def test_near_a_point_and_when_the_ring_is_crowded():
    floor = room_floor()
    mug = fp("mug", 0.08, 0.08, 0.1)
    r = P.place(floor, mug, [], P.Rule(at=P.Near((0.5, 0.5), 0.3, 0.0)))
    assert P.gap_to_point(r.pose.prism.rect, (0.5, 0.5)) == pytest.approx(0.3, abs=1e-6)
    assert r.pose.prism.rect.centre == pytest.approx([0.5, 0.5 - 0.3 - 0.04])         # straight ahead of a front facing -Y
    blocker = prism("wall of crates", 0.5, 0.0, 0.5, 0.3, 0.0, 1.0)                    # sits where bearing 0 would land
    r = P.place(floor, mug, [blocker], P.Rule(at=P.Near((0.5, 0.5), 0.3, 0.0)))
    assert P.gap_to_point(r.pose.prism.rect, (0.5, 0.5)) == pytest.approx(0.3, abs=1e-6)
    assert r.tries > 1 and abs(r.pose.uv[1] - 0.5) < 0.5                              # next bearing round the ring
    a = P.place(floor, mug, [blocker], P.Rule(at=P.Near((0.5, 0.5), 0.3, 0.0), seed=1))
    b = P.place(floor, mug, [blocker], P.Rule(at=P.Near((0.5, 0.5), 0.3, 0.0), seed=1))
    assert a.pose.uv == b.pose.uv                                                      # seeded
    tiny = P.Surface("pad", "floor", (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.1, 0.1))
    with pytest.raises(P.PlaceError, match="candidates tried"):
        P.place(tiny, mug, [], P.Rule(at=P.Near((0.0, 0.0), 0.5, 0.0)))


# ---------------------------------------------------------------- regions, walls, edges
def test_region_is_seeded_and_stays_clear():
    top = table()
    box = fp("box", 0.1, 0.1, 0.1)
    others = [prism("lamp", 0.0, 0.0, 0.1, 0.1, 0.74, 1.0)]
    rule = P.Rule(at=P.Region((-0.5, -0.2), (0.5, 0.2)), seed=7)
    a, b = P.place(top, box, others, rule), P.place(top, box, others, rule)
    assert a.pose.uv == b.pose.uv
    assert P.place(top, box, others, P.Rule(at=P.Region((-0.5, -0.2), (0.5, 0.2)), seed=8)).pose.uv != a.pose.uv
    u, v = a.pose.uv
    assert -0.5 <= u <= 0.5 and -0.2 <= v <= 0.2 and P.gap(a.pose.prism.rect, others[0].rect) >= 0.02 - 1e-9
    with pytest.raises(P.PlaceError, match="400 candidates tried"):
        P.place(top, box, others, P.Rule(at=P.Region((-0.05, -0.05), (0.05, 0.05))))


def test_posters_on_a_wall_with_the_window_in_the_way():
    wall = P.surface_from_entry("wall_back", {"center": [0, 1.6, 1.275], "normal": [0, -1, 0], "size": [3.6, 2.55]},
                                np.eye(4))
    poster = P.Footprint("poster", (-0.25, -0.004, -0.35), (0.25, 0.0, 0.35), wall_origin=True)
    window = prism("window", 0.4, 1.55, 0.65, 0.05, 0.95, 2.15)
    r = P.place(wall, poster, [window], P.Rule(at=(-1.2, 0.4)))
    assert r.pose.origin == pytest.approx([-1.2, 1.6, 1.675]) and r.yaw_deg == pytest.approx(0.0)
    with pytest.raises(P.PlaceError, match="overlaps 'window'"):
        P.place(wall, poster, [window], P.Rule(at=(0.4, 0.4)))
    with pytest.raises(P.PlaceError, match="faces out of it"):
        P.place(wall, poster, [], P.Rule(at=(0.0, 0.0), facing=("yaw", 30)))
    with pytest.raises(P.PlaceError, match="too close to the top edge"):
        P.place(wall, poster, [], P.Rule(at=(0.0, 1.0)))                             # 1.0 + 0.35 > 1.275 - 0.02
    r = P.place(wall, poster, [], P.Rule(at=(0.0, 0.0), align=("top", "right"), clear=0.05))
    assert r.pose.uv == pytest.approx((1.8 - 0.05 - 0.25, 1.275 - 0.05 - 0.35))


def test_an_edge_surface_places_along_the_line():
    edge = P.surface_from_entry("rim", {"type": "edge", "a": [-0.14, 0.28, 0.74], "b": [0.14, 0.28, 0.74],
                                        "normal": [0, 0, 1]}, np.eye(4))
    pen = fp("pen", 0.14, 0.012, 0.012)
    r = P.place(edge, pen, [], P.Rule(at=(0.05,)))
    assert r.pose.uv == (0.05, 0.0) and r.pose.origin == pytest.approx([0.05, 0.28, 0.74])
    with pytest.raises(P.PlaceError, match="end of the edge"):
        P.place(edge, pen, [], P.Rule(at=(0.2,)))
    with pytest.raises(P.PlaceError, match="no sides"):
        P.place(edge, pen, [], P.Rule(align=("front",)))


# ---------------------------------------------------------------- scatter
def test_scatter_counts_clears_and_is_deterministic():
    top = table()
    tapes = [fp(f"tape_{i}", 0.1, 0.064, 0.0125) for i in range(5)]
    lamp = prism("lamp", -0.4, 0.0, 0.08, 0.08, 0.74, 1.2)
    kw = dict(region=((0.0, -0.2), (0.5, 0.2)), min_dist=0.03, yaw=(-25, 25), clear=0.02, seed=3)
    a = P.scatter(top, tapes, [lamp], **kw)
    b = P.scatter(top, tapes, [lamp], **kw)
    assert [x.pose.uv for x in a] == [x.pose.uv for x in b] and [x.yaw_deg for x in a] == [x.yaw_deg for x in b]
    c = P.scatter(top, tapes, [lamp], **dict(kw, seed=4))
    assert [x.pose.uv for x in c] != [x.pose.uv for x in a]
    assert len(a) == 5
    for i, x in enumerate(a):
        u, v = x.pose.uv
        assert 0.0 <= u <= 0.5 and -0.2 <= v <= 0.2 and -25.0 <= x.yaw_deg <= 25.0
        assert P.gap(x.pose.prism.rect, lamp.rect) >= 0.02 - 1e-9
        for y in a[:i]:
            assert P.gap(x.pose.prism.rect, y.pose.prism.rect) >= 0.03 - 1e-9              # min_dist between scattered
        assert x.pose.origin[2] == pytest.approx(0.74)
    assert [x.name for x in P.scatter(top, tapes[:2], [], names=["a", "b"], seed=1)] == ["a", "b"]


def test_scatter_fails_with_a_count_and_a_reason_when_the_region_is_full():
    top = table()
    big = [fp(f"b{i}", 0.3, 0.3, 0.1) for i in range(6)]
    with pytest.raises(P.PlaceError) as e:
        P.scatter(top, big, [], region=((-0.3, -0.1), (0.3, 0.1)), min_dist=0.05, seed=1, tries=60)
    msg = str(e.value)
    assert msg.startswith("scatter: placed ") and " of 6;" in msg and "found no spot in 60 tries" in msg


def test_scatter_on_a_wall_ignores_yaw_and_respects_obstacles():
    wall = P.surface_from_entry("wall_left", {"center": [-1.8, 0, 1.275], "normal": [1, 0, 0], "size": [3.2, 2.55]},
                                np.eye(4))
    posters = [P.Footprint(f"p{i}", (-0.2, -0.004, -0.3), (0.2, 0.0, 0.3), wall_origin=True) for i in range(3)]
    door = prism("door", -1.75, 1.0, 0.05, 0.3, 0.0, 2.05)
    out = P.scatter(wall, posters, [door], region=((-1.4, -0.2), (1.4, 0.5)), min_dist=0.1, yaw=(-45, 45), seed=2)
    assert len(out) == 3 and all(abs(o.yaw_deg - 90.0) < 1e-6 for o in out)
    assert all(P.gap(o.pose.prism.rect, door.rect) >= 0.02 - 1e-9 for o in out)


def test_scatter_on_a_round_top_uses_the_whole_disc():
    top = P.surface_from_entry("top", {"type": "plane", "center": [0, 0, 0.74], "normal": [0, 0, 1], "radius": 0.32},
                               np.eye(4))
    cups = [fp(f"c{i}", 0.06, 0.06, 0.08) for i in range(6)]
    out = P.scatter(top, cups, [], seed=1)
    assert len(out) == 6 and max(abs(o.pose.uv[1]) for o in out) > 0.1               # v is not squeezed to the line v = 0
    assert all(max(np.linalg.norm(o.pose.corners, axis=1)) <= 0.32 - 0.02 + 1e-9 for o in out)


def test_the_host_of_a_surface_never_blocks_what_stands_on_it():
    # a bed whose headboard rises above the quilt: its bounds reach z 0.95 but the quilt plane is at 0.55
    quilt = P.surface_from_entry("bed:top", {"center": [0, 0, 0.55], "normal": [0, 0, 1], "size": [0.9, 1.5]}, np.eye(4))
    bed = prism("bed", 0, 0, 0.5, 1.0, 0.0, 0.95)
    book = fp("book", 0.15, 0.2, 0.03)
    with pytest.raises(P.PlaceError, match="overlaps 'bed'"):
        P.place(quilt, book, [bed], P.Rule(at=(0.0, 0.0)))
    assert P.place(quilt, book, [bed], P.Rule(at=(0.0, 0.0), host="bed")).pose.origin[2] == pytest.approx(0.55)
    with pytest.raises(P.PlaceError, match="overlaps 'bed'"):                         # unless the rule names it
        P.place(quilt, book, [bed], P.Rule(at=(0.0, 0.0), host="bed", avoid=("bed",)))
    out = P.scatter(quilt, [fp(f"b{i}", 0.15, 0.2, 0.03) for i in range(3)], [bed], seed=2, host="bed")
    assert len(out) == 3


# ---------------------------------------------------------------- height layers
def grid(x0, x1, y0, y1, z0, z1, n=6):
    xs, ys, zs = (np.linspace(a, b, n) for a, b in ((x0, x1), (y0, y1), (z0, z1)))
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])


def test_layers_of_points_follow_the_shape_of_a_chair():
    seat = grid(-0.2, 0.2, -0.22, 0.25, 0.0, 0.45, n=10)                  # legs and seat: the whole footprint up to 0.45
    back = grid(-0.19, 0.19, 0.17, 0.25, 0.45, 0.88, n=10)                # the backrest above it
    L = P.layers_of_points(np.concatenate([seat, back]), step=0.1, merge=0.005)
    assert L[0][0] == pytest.approx(0.0) and L[-1][1] == pytest.approx(0.88)
    low, high = L[0], L[-1]
    assert low[2:] == pytest.approx((-0.2, -0.22, 0.2, 0.25))               # wide at the bottom
    assert high[3] == pytest.approx(0.17) and high[5] == pytest.approx(0.25) and high[2] == pytest.approx(-0.19)
    assert all(a[1] <= b[0] + 1e-9 for a, b in zip(L, L[1:]))                  # ordered, not overlapping in z
    assert len(L) >= 2 and P.layers_of_points(np.zeros((0, 3))) == []
    assert len(P.layers_of_points(grid(0, 1, 0, 1, 0, 0.012), step=0.05)) == 1  # a thin thing is one layer


def test_prisms_of_turns_every_layer_with_the_prop():
    layers = ((0.0, 0.45, -0.2, -0.22, 0.2, 0.25), (0.45, 0.88, -0.19, 0.17, 0.19, 0.25))
    ps = P.prisms_of("chair", (-0.2, -0.22, 0.0), (0.2, 0.25, 0.88), rot_z(90, (1.0, 2.0, 0.1)), layers=layers)
    assert len(ps) == 2 and all(p.rect.yaw == pytest.approx(math.pi / 2) for p in ps)
    assert ps[1].rect.centre == pytest.approx([1.0 - 0.21, 2.0])                  # the backrest's (0, 0.21) turned by 90 deg
    assert (ps[1].z0, ps[1].z1) == pytest.approx((0.55, 0.98))
    assert len(P.prisms_of("chair", (-0.2, -0.22, 0.0), (0.2, 0.25, 0.88), rot_z(0), layers=())) == 1


def test_a_lamp_on_a_desk_clears_the_chair_tucked_under_it_and_a_mug_fits_under_the_shade():
    desk = prism("desk", 0.4, 1.28, 0.6, 0.3, 0.0, 0.74)
    # the chair faces +Y: its seat/legs reach 0.06 m under the desk, its tall backrest stays well in front of it
    chair = [P.Prism("chair", P.Rect(0.4, 0.95, 0.2, 0.25), 0.0, 0.45),
             P.Prism("chair", P.Rect(0.4, 0.70, 0.19, 0.04), 0.45, 0.88)]
    top = P.surface_from_entry("desk:top", {"center": [0.4, 1.28, 0.74], "normal": [0, 0, 1], "size": [1.2, 0.6]}, np.eye(4))
    # a lamp: a small base, a long shade at 0.3-0.45 reaching 0.4 m toward the front (-Y)
    lamp = P.Footprint("lamp", (-0.08, -0.4, 0.0), (0.08, 0.11, 0.45),
                       layers=((0.0, 0.05, -0.08, -0.08, 0.08, 0.11), (0.05, 0.3, -0.04, -0.05, 0.04, 0.11),
                               (0.3, 0.45, -0.07, -0.4, 0.07, 0.0)))
    others = [desk, *chair]
    one_box = P.Footprint("lamp", lamp.lo, lamp.hi)
    with pytest.raises(P.PlaceError, match="chair"):                              # as one box its shade overlaps the chair
        P.place(top, one_box, [chair[0], chair[1], P.Prism("chair", P.Rect(0.4, 0.95, 0.2, 0.25), 0.0, 0.88)],
                P.Rule(at=(0.1, 0.0), align=("back",), clear=0.04, host="desk"))
    r = P.place(top, lamp, others, P.Rule(at=(0.1, 0.0), align=("back",), clear=0.04, host="desk"))
    assert r.pose.uv[1] == pytest.approx(0.3 - 0.04 - 0.255) and len(r.pose.layers) == 3     # footprint centre
    # a mug (0.1 tall) under the shade is fine; a 0.5 m tall vase there is blocked by the shade
    mug, vase = fp("mug", 0.08, 0.08, 0.1), fp("vase", 0.08, 0.08, 0.5)
    placed = r.pose.layers
    under = P.Rule(at=(0.1, 0.05 - 0.2), clear=0.02, host="desk")                  # in front of the lamp base, below the shade
    assert P.place(top, mug, others + placed, under).pose.origin[2] == pytest.approx(0.74)
    with pytest.raises(P.PlaceError, match="overlaps 'lamp'"):
        P.place(top, vase, others + placed, under)
    # scatter keeps every layer of what it places
    tall = [fp(f"t{i}", 0.1, 0.1, 0.5) for i in range(2)]
    out = P.scatter(top, tall, others + placed, region=((0.1, -0.2), (0.5, 0.25)), seed=1, host="desk")
    assert len(out) == 2 and all(len(o.pose.layers) == 1 for o in out)
