"""Automatic prop cards without Blender: origin from bounds, the flat top face (boxes, round tops, hollow and sloped shapes,
a table with things on it), moving parts, colliders from rigid bodies, the card and merging a project's extras."""
import math

import numpy as np
import pytest

from mkmmd.core import propcard as PC


def box_tris(lo, hi):
    """12 triangles of an axis-aligned box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
         (3, 0, 4), (3, 4, 7)]
    return v[np.array(f)]


def cylinder_tris(r, z0, z1, n=48, cx=0.0, cy=0.0, inner=None):
    """A closed cylinder (or a tube with an open top when `inner` is a radius: a floor at z0 + 0.005 inside)."""
    a = 2 * math.pi * np.arange(n) / n
    ring = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)

    def v(p, z):
        return (p[0], p[1], z)

    out = []
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]
        out += [[v(p, z0), v(q, z0), v(q, z1)], [v(p, z0), v(q, z1), v(p, z1)]]
        if inner is None:
            out += [[(cx, cy, z1), v(p, z1), v(q, z1)], [(cx, cy, z0), v(q, z0), v(p, z0)]]
        else:
            ip, iq = ring[i] * inner / r, ring[(i + 1) % n] * inner / r
            out += [[v(p, z1), v(q, z1), v(iq, z1)], [v(p, z1), v(iq, z1), v(ip, z1)],
                    [(cx, cy, z0 + 0.005), v(ip, z0 + 0.005), v(iq, z0 + 0.005)],
                    [(cx, cy, z0), v(q, z0), v(p, z0)]]
    return np.array(out, float)


# ---------------------------------------------------------------- bounds and origin
def test_bounds_and_origin_shift():
    t = box_tris((1.0, 2.0, 0.5), (1.4, 2.6, 0.9))
    lo, hi = PC.bounds(t)
    assert lo.tolist() == [1.0, 2.0, 0.5] and hi.tolist() == [1.4, 2.6, 0.9]
    assert PC.origin_shift(lo, hi).tolist() == pytest.approx([-1.2, -2.3, -0.5])               # floor_center
    assert PC.origin_shift(lo, hi, "center").tolist() == pytest.approx([-1.2, -2.3, -0.7])
    assert PC.origin_shift(lo, hi, "keep").tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="origin must be one of"):
        PC.origin_shift(lo, hi, "corner")
    with pytest.raises(ValueError, match="no triangles"):
        PC.bounds(np.zeros((0, 3, 3)))


# ---------------------------------------------------------------- the top face
def test_a_box_is_flat_topped():
    top = PC.top_surface(box_tris((-0.7, -0.25, 0.0), (0.7, 0.25, 0.94)))
    assert top["z"] == pytest.approx(0.94) and top["size"] == pytest.approx([1.4, 0.5], abs=0.03)
    assert top["center"] == pytest.approx([0.0, 0.0], abs=0.02) and top["fill"] > 0.95 and top["share"] > 0.95
    assert "radius" not in top


def test_a_round_top_gets_a_radius():
    top = PC.top_surface(cylinder_tris(0.32, 0.0, 0.74))
    assert top["z"] == pytest.approx(0.74) and top["radius"] == pytest.approx(0.32, abs=0.02)
    assert 0.7 < top["fill"] < 0.86


def test_hollow_and_sloped_shapes_are_not_flat_topped():
    cup = cylinder_tris(0.05, 0.0, 0.078, inner=0.045)
    assert PC.top_surface(cup) is None                                     # the rim is thin; the floor is far below it
    ramp = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0.5]], [[0, 0, 0], [1, 1, 0.5], [0, 1, 0.5]],
                     [[0, 0, 0], [0, 1, 0.5], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [1, 1, 0.5]]], float)
    assert PC.top_surface(ramp) is None                                    # heights spread over the whole range
    dome = np.array([[[0, 0, 0.0], [1, 0, 0.0], [0.5, 0.5, 1.0]], [[1, 0, 0], [1, 1, 0], [0.5, 0.5, 1.0]],
                     [[1, 1, 0], [0, 1, 0], [0.5, 0.5, 1.0]], [[0, 1, 0], [0, 0, 0], [0.5, 0.5, 1.0]]], float)
    assert PC.top_surface(dome) is None


def test_a_table_with_a_book_has_a_top_but_with_a_lamp_it_does_not():
    desk = box_tris((-0.6, -0.3, 0.0), (0.6, 0.3, 0.74))
    book = box_tris((-0.1, -0.1, 0.74), (0.1, 0.1, 0.77))
    top = PC.top_surface(np.concatenate([desk, book]))
    assert top["z"] == pytest.approx(0.74, abs=0.004) and top["size"] == pytest.approx([1.2, 0.6], abs=0.03)
    lamp = cylinder_tris(0.06, 0.74, 1.19, cx=0.3)
    assert PC.top_surface(np.concatenate([desk, lamp])) is None            # the highest point is 45 cm above the top


def test_a_thin_flat_sheet_is_a_top():
    sheet = np.array([[[-1, -1, 0], [1, -1, 0], [1, 1, 0]], [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]]], float)
    top = PC.top_surface(sheet)                                            # two big triangles: sampled by subdivision
    assert top["z"] == pytest.approx(0.0) and top["size"] == pytest.approx([2.0, 2.0], abs=0.05) and top["fill"] > 0.95


# ---------------------------------------------------------------- moving parts
def test_moving_parts_by_bone_weights():
    # a rigid prop: everything on one bone, or a bone with three stray vertices
    assert PC.moving_parts({"root": None}, {"root": 900}) == []
    assert PC.moving_parts({"root": None, "b": "root"}, {"root": 900, "b": 3}) == []
    # a base and a part under it
    assert PC.moving_parts({"base": None, "cup": "base"}, {"base": 207, "cup": 794}) == ["cup"]
    # a mixer: faders and knobs under a base that carries most of the body
    parents = {"all": None, "fader1": "all", "fader2": "all", "knob": "all"}
    assert PC.moving_parts(parents, {"all": 722, "fader1": 92, "fader2": 92, "knob": 81}) == ["fader1", "fader2", "knob"]
    # sibling carriers under an unweighted root are all parts
    parents = {"all": None, "1": "all", "2": "all", "3": "all"}
    assert PC.moving_parts(parents, {"1": 64, "2": 41, "3": 35}) == ["1", "2", "3"]
    assert PC.moving_parts({}, {}) == []


# ---------------------------------------------------------------- colliders
def test_colliders_from_static_rigid_bodies_skip_the_soft_ones():
    R = np.eye(3)
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)               # turned 90 deg about z
    bodies = [{"shape": "SPHERE", "mode": 0, "center": [0, 0, 0.2], "rot": R, "half": [0.04, 0.04, 0.04]},
              {"shape": "CAPSULE", "mode": 0, "center": [0.1, 0, 0.1], "rot": Rz, "half": [0.01, 0.05, 0.01]},
              {"shape": "BOX", "mode": 0, "center": [0, 0, 0.05], "rot": R, "half": [0.1, 0.05, 0.05]},
              {"shape": "CAPSULE", "mode": 2, "center": [0, 0, 0.3], "rot": R, "half": [0.01, 0.01, 0.05]}]
    shapes = PC.shapes_from_bodies(bodies)
    assert [s["kind"] for s in shapes] == ["sphere", "capsule", "box"]       # the dynamic cord is left out
    assert shapes[0]["R"] == pytest.approx(0.04)
    cap = shapes[1]                                                         # long axis y, turned to -x... by 90 deg: along x
    assert cap["R"] == pytest.approx(0.01)
    assert cap["a"] == pytest.approx([0.1 + 0.04, 0.0, 0.1]) or cap["a"] == pytest.approx([0.1 - 0.04, 0.0, 0.1])
    assert abs(cap["b"][0] - cap["a"][0]) == pytest.approx(0.08) and cap["a"][1] == pytest.approx(0.0)
    assert PC.shapes_from_bodies(bodies, static_only=False)[3]["kind"] == "capsule"
    assert len(PC.shapes_from_bodies(bodies, static_only=False)) == 4


def test_one_box_over_the_bounds_has_a_minimum_thickness():
    s = PC.box_shape((-0.5, -0.5, -0.004), (0.5, 0.5, -0.004))               # a cloth: zero thickness
    assert s["half"] == pytest.approx([0.5, 0.5, 0.005]) and s["center"] == pytest.approx([0.0, 0.0, -0.004])


# ---------------------------------------------------------------- the card
def test_make_card_of_a_desk_with_a_flat_top_and_no_bodies():
    t = box_tris((-0.7, -0.25, 0.0), (0.7, 0.25, 0.94))
    card, shapes = PC.make_card("desk", "desk.pmx", t)
    assert card["kind"] == "pmx" and card["origin"] == "floor_center" and card["front"] == "-Y"
    assert card["size"] == pytest.approx([1.4, 0.5, 0.94]) and card["bounds"]["min"] == pytest.approx([-0.7, -0.25, 0.0])
    assert card["use"]["look"] == [{"name": "center", "point": [0.0, 0.0, 0.47]}]
    top = card["use"]["rest"][0]
    assert top["name"] == "top" and top["type"] == "plane" and top["normal"] == [0, 0, 1]
    assert top["center"][2] == pytest.approx(0.94, abs=0.004) and top["size"] == pytest.approx([1.4, 0.5], abs=0.03)
    assert len(shapes) == 1 and shapes[0]["kind"] == "box" and card["stats"]["colliders_from"] == "bounds"
    assert card["parts"] == [] and card["armature"] is None and card["slots"] == {}


def test_make_card_of_a_dome_has_no_rest_plane_and_takes_the_bodies():
    dome = np.array([[[0, 0, 0.0], [1, 0, 0.0], [0.5, 0.5, 1.0]], [[1, 0, 0], [1, 1, 0], [0.5, 0.5, 1.0]],
                     [[1, 1, 0], [0, 1, 0], [0.5, 0.5, 1.0]], [[0, 1, 0], [0, 0, 0], [0.5, 0.5, 1.0]]], float)
    body = {"shape": "SPHERE", "mode": 0, "center": [0.5, 0.5, 0.5], "rot": np.eye(3), "half": [0.4, 0.4, 0.4]}
    card, shapes = PC.make_card("blob", "b.pmx", dome, bodies=[body], parts=["ear"], armature="blob_arm")
    assert "rest" not in card["use"] and [s["kind"] for s in shapes] == ["sphere"]
    assert card["stats"]["colliders_from"] == "rigid bodies" and card["parts"] == ["ear"] and card["armature"] == "blob_arm"
    assert PC.describe_shape(shapes[0]) == "sphere R 0.400"


def test_merge_extra_overrides_extends_and_clears():
    card = {"name": "p", "size": [1, 2, 3], "use": {"rest": [{"name": "top", "center": [0, 0, 1]}],
                                                     "look": [{"name": "center", "point": [0, 0, 0.5]}]},
            "colliders": [{"type": "box", "object": "p_col"}], "front": "-Y"}
    extra = {"front": "+X", "use": {"rest": [{"name": "top", "center": [0, 0, 2], "size": [1, 1]},
                                              {"name": "edge", "type": "edge", "a": [0, 0, 0], "b": [1, 0, 0]}],
                                    "grip": [{"name": "handle", "type": "pen"}]},
             "colliders": [], "license": {"credit": "x"}}
    out = PC.merge_extra(card, extra)
    assert out["front"] == "+X" and out["colliders"] == [] and out["size"] == [1, 2, 3]
    assert [e["name"] for e in out["use"]["rest"]] == ["top", "edge"] and out["use"]["rest"][0]["center"] == [0, 0, 2]
    assert out["use"]["look"] == card["use"]["look"] and out["use"]["grip"][0]["name"] == "handle"
    assert out["license"] == {"credit": "x"}
    assert card["use"]["rest"][0]["center"] == [0, 0, 1]                      # the original is untouched
    out["use"]["look"][0]["point"][0] = 9
    assert card["use"]["look"][0]["point"][0] == 0
    assert PC.merge_extra(card, None) is card
