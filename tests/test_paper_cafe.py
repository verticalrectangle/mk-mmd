"""The café's chair and table of card (mkmmd.core.paper_cafe, no bpy): she sits on the chair's seat with its back behind
her and its legs on the floor, the scissors lie on the table clear of the cup, the lily pad floats on the coffee, the
cup jumps on its rattle and settles, and every piece is a closed solid."""
import math

import numpy as np

from mkmmd.blender.library.sets import paper_theatre_parts as PTP
from mkmmd.core import paper_cafe as PC
from mkmmd.core import paper_theatre as PT
from mkmmd.core import shell as S


def _mesh(pieces):
    return S.merge([PTP.plate_mesh(p._replace(role="cardboard", edge=None), k) if isinstance(p, PT.Plate) else p
                    for k, p in enumerate(pieces)])


def test_she_sits_on_the_seat_her_feet_on_the_floor_the_back_behind_her():
    card = PC.chair_card("chair", PC.CHAIR_ROLES, [])
    sit, feet = card["use"]["sit"][0], card["use"]["feet"][0]
    assert sit["name"] == feet["name"] == "seat" and sit["facing"] == [0, -1, 0] and sit["floor_z"] == 0.0
    assert math.isclose(sit["hip"][2] - PC.SEAT_Z, 0.075) and abs(sit["hip"][0]) < 1e-9
    P = PC.chair_pieces()
    body, seat = _mesh(P["body"]), _mesh(P["seat"])
    assert abs(body.V[:, 2].min()) < 0.01                                      # the legs stand on the floor
    top = seat.V[np.hypot(seat.V[:, 0], seat.V[:, 1]) < 0.05, 2].max()
    assert PC.SEAT_Z - 0.002 < top < PC.SEAT_Z + 0.008                         # the seat's top where her hips go
    back = PC.back_path()
    high = back[back[:, 2] > PC.SEAT_Z + 0.1]
    assert (high[:, 1] > sit["hip"][1] + 0.08).all() and high[:, 2].max() > 0.8   # behind her, up to her shoulders
    assert feet["L"][1] < -0.25 and feet["R"][1] < -0.25 and feet["L"][0] > 0 > feet["R"][0]


def test_the_scissors_lie_on_the_table_clear_of_the_cup_and_the_pad_floats_in_it():
    card = PC.table_card("table", PC.table_options({}), PC.TABLE_ROLES, "table_cup", [])
    a = card["use"]["anchor"][0]
    x, y, z = a["point"]
    assert math.hypot(x, y) + 0.07 < PC.TOP_R and abs(z - PC.TOP_Z) < 0.01      # on the top, 7 cm of blade inside
    assert math.hypot(x - PC.CUP_AT[0], y - PC.CUP_AT[1]) > PC.SAUCER[0] + 0.07
    assert x > 0.0                                                               # on her side (she sits at +x)
    P = PC.table_pieces()
    cup, coffee, pad = P["cup"][0], P["cup"][2], P["cup"][3]
    r0, r1, h, t = PC.CUP
    assert coffee.V[:, 2].max() < h and np.hypot(coffee.V[:, 0], coffee.V[:, 1]).max() < r1 - t
    assert pad.V[:, 2].min() > PC.COFFEE - 0.0005 and np.hypot(pad.V[:, 0], pad.V[:, 1]).max() < r1 - t
    table = _mesh(P["table"])
    assert abs(table.V[:, 2].min()) < 0.01 and abs(table.V[:, 2].max() - PC.TOP_Z) < 0.01


def test_the_cup_rattles_on_its_time_and_settles():
    o = PC.table_options({"rattle": 30.15})
    t = np.arange(29.0, 31.12, 1.0 / 60.0)
    cz, tx, _ty, sz = PC.rattle(o, t)
    before, after = t < 30.15, t >= 30.15
    assert np.all(cz[before] == 0.0) and np.all(sz[before] == 0.0) and np.all(tx[before] == 0.0)
    assert cz[after & (t < 30.3)].max() > 0.005                                 # it jumps
    assert np.abs(cz[t > 30.15 + PC.RATTLE[0]]).max() < 0.001                   # and is still half a second on


def test_every_piece_is_a_closed_outward_solid():
    P, T = PC.chair_pieces(), PC.table_pieces()
    for name, pieces in (("body", P["body"]), ("seat", P["seat"]), ("table", T["table"]), ("saucer", T["saucer"]),
                         ("cup", T["cup"])):
        for k, p in enumerate(pieces):
            m = _mesh([p])
            assert m.is_closed() and m.volume() > 0, (name, k)
