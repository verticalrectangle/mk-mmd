"""Basic props: a bentwood cafe chair and an open two-seat car cabin mock-up."""
import math

import bmesh

from ..mesh import box, cylinder, material, torus
from . import register


def _round_edges(o, width, segments=3, rim=False):
    """Fillets instead of hard edges on a primitive: bevel its edges `width` m wide in `segments` steps (only the rims of
    its n-gon caps when `rim`), the new faces smooth shaded. Hard edges catch no light (docs/modelling.md)."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    edges = [e for e in bm.edges if not rim or any(len(f.verts) > 4 for f in e.link_faces)]
    made = bmesh.ops.bevel(bm, geom=edges, offset=width, offset_type="OFFSET", segments=segments, profile=0.5,
                           affect="EDGES")
    for f in made["faces"]:
        f.smooth = True
    bm.to_mesh(o.data)
    bm.free()
    return o


@register("chair")
def chair(name, coll, root, slots=None):
    """A bentwood cafe chair: round seat (top 0.45 m, radius 0.205), four legs, a curved backrest."""
    s = dict({"wood": "#b4637a", "cane": "#ea9d34"}, **(slots or {}))
    wood, cane = material(f"{name}_wood", s["wood"], 0.45), material(f"{name}_cane", s["cane"], 0.7)
    seat_z, r = 0.45, 0.205
    _round_edges(cylinder(f"{name}_seat", (0, 0, seat_z - 0.02), r, 0.04, coll, root, mat=cane), 0.014, 4, rim=True)
    for i, a in enumerate((45, 135, 225, 315)):
        x, y = 0.15 * math.cos(math.radians(a)), 0.15 * math.sin(math.radians(a))
        cylinder(f"{name}_leg{i}", (x, y, (seat_z - 0.04) / 2), 0.014, seat_z - 0.04, coll, root, mat=wood, segments=12)
    for i, x in enumerate((-0.17, 0.17)):
        cylinder(f"{name}_post{i}", (x, 0.17, seat_z + 0.22), 0.012, 0.44, coll, root, rot=(-8, 0, 0), mat=wood,
                 segments=12)
    _round_edges(box(f"{name}_backrail", (0, 0.20, seat_z + 0.40), (0.36, 0.03, 0.06), coll, root, rot=(-8, 0, 0),
                     mat=wood), 0.012, 3)
    cols = [box(f"{name}_col_back", (0, 0.19, seat_z + 0.30), (0.38, 0.05, 0.30), coll, root, rot=(-8, 0, 0),
                collider=True),
            cylinder(f"{name}_col_seat", (0, 0, seat_z - 0.02), r, 0.04, coll, root, collider=True)]
    return {
        "size": [0.43, 0.48, 0.90], "origin": "floor_center", "front": "-Y", "slots": s,
        "use": {
            "sit": [{"name": "seat", "hip": [0.0, 0.035, seat_z + 0.075], "facing": [0, -1, 0], "seat_z": seat_z,
                     "floor_z": 0.0, "pelvis_deg": 6.0, "back_deg": 0.0}],
            "feet": [{"name": "floor", "L": [0.115, -0.33, None], "R": [-0.105, -0.36, None], "floor_z": 0.0}],
        },
        "colliders": [{"type": "box", "object": cols[0].name, "rnd": 0.012, "tag": name},
                      {"type": "cylinder", "object": cols[1].name, "half_h": 0.02, "rnd": 0.012, "tag": name}],
    }


@register("car_mockup")
def car_mockup(name, coll, root, slots=None):
    """An open two-seat cabin: floor, two bucket seats, dashboard, doors with sills, a steering wheel on the left (+X)
    seat. Forward is -Y. Enough to seat two characters, steer, rest an arm on a door and test wind."""
    s = dict({"body": "#3e8fb0", "seat": "#393552", "trim": "#e0def4", "dash": "#2a273f"}, **(slots or {}))
    body, seat_m = material(f"{name}_body", s["body"], 0.35, 0.3), material(f"{name}_seat", s["seat"], 0.8)
    trim, dash = material(f"{name}_trim", s["trim"], 0.3), material(f"{name}_dash", s["dash"], 0.6)
    floor_z, seat_top = 0.30, 0.58
    xs = {"driver": 0.38, "passenger": -0.38}
    cols = [box(f"{name}_floor", (0, 0, floor_z - 0.03), (1.46, 2.2, 0.06), coll, root, mat=body)]
    box(f"{name}_tub", (0, 0.1, (floor_z - 0.06) / 2), (1.6, 3.6, floor_z - 0.06), coll, root, mat=body)
    for side, x in xs.items():
        box(f"{name}_cushion_{side}", (x, 0.18, seat_top - 0.06), (0.50, 0.50, 0.12), coll, root, mat=seat_m)
        box(f"{name}_back_{side}", (x, 0.50, seat_top + 0.30), (0.50, 0.10, 0.62), coll, root, rot=(18, 0, 0),
            mat=seat_m)
        cols.append(box(f"{name}_col_cushion_{side}", (x, 0.18, seat_top - 0.06), (0.50, 0.50, 0.12), coll, root,
                        collider=True))
        cols.append(box(f"{name}_col_back_{side}", (x, 0.50, seat_top + 0.30), (0.50, 0.10, 0.62), coll, root,
                        rot=(18, 0, 0), collider=True))
    box(f"{name}_dash", (0, -0.62, 0.86), (1.46, 0.36, 0.24), coll, root, mat=dash)
    cols.append(box(f"{name}_col_dash", (0, -0.62, 0.86), (1.46, 0.36, 0.24), coll, root, collider=True))
    for side, x in (("L", 0.76), ("R", -0.76)):
        box(f"{name}_door_{side}", (x, -0.05, 0.62), (0.08, 1.30, 0.64), coll, root, mat=body)
        box(f"{name}_sill_{side}", (x, -0.05, 0.95), (0.10, 1.30, 0.03), coll, root, mat=trim)
        cols.append(box(f"{name}_col_door_{side}", (x, -0.05, 0.64), (0.08, 1.30, 0.66), coll, root, collider=True))
    box(f"{name}_windshield_frame", (0, -0.80, 1.18), (1.46, 0.04, 0.04), coll, root, mat=trim)
    wheel_c, wheel_axis, wheel_r, wheel_t = (xs["driver"], -0.30, 0.95), (0.0, 1.0, 0.55), 0.19, 0.016
    torus(f"{name}_wheel", wheel_c, wheel_axis, wheel_r, wheel_t, coll, root, mat=dash)
    cylinder(f"{name}_column", (xs["driver"], -0.42, 0.88), 0.03, 0.30, coll, root, rot=(61, 0, 0), mat=dash,
             segments=12)
    tyre = material(f"{name}_tyre", s["seat"], 0.9)
    road_wheels = [cylinder(f"{name}_roadwheel_{k}", (x, y, 0.30), 0.30, 0.22, coll, root, rot=(0, 90, 0), mat=tyre,
                            segments=24)
                   for k, (x, y) in enumerate(((0.82, -1.25), (-0.82, -1.25), (0.82, 1.20), (-0.82, 1.20)))]
    sits, feet = [], []
    for side, x in xs.items():
        sits.append({"name": side, "hip": [x, 0.22, seat_top + 0.07], "facing": [0, -1, 0], "seat_z": seat_top,
                     "floor_z": floor_z, "pelvis_deg": 10.0, "back_deg": -14.0})
        feet.append({"name": side, "L": [x + 0.10, -0.48, None], "R": [x - 0.10, -0.50, None], "floor_z": floor_z})
    return {
        "size": [1.6, 3.6, 1.2], "origin": "floor_center", "front": "-Y", "slots": s,
        "use": {
            "sit": sits, "feet": feet,
            "grip": [{"name": "wheel", "type": "ring", "center": list(wheel_c), "axis": list(wheel_axis),
                      "radius": wheel_r, "tube": wheel_t, "object": f"{name}_wheel"}],
            "rest": [{"name": "sill_L", "type": "edge", "a": [0.76, -0.6, 0.97], "b": [0.76, 0.5, 0.97],
                      "normal": [0, 0, 1]},
                     {"name": "sill_R", "type": "edge", "a": [-0.76, -0.6, 0.97], "b": [-0.76, 0.5, 0.97],
                      "normal": [0, 0, 1]}],
            "look": [{"name": "road", "point": [0.0, -30.0, 1.2]}, {"name": "mirror", "point": [0.0, -0.75, 1.30]}],
        },
        "colliders": [{"type": "box", "object": o.name, "rnd": 0.01, "tag": name} for o in cols]
        + [{"type": "ring", "object": f"{name}_wheel", "radius": wheel_r, "tube": wheel_t, "tag": name}],
        "wheels": [{"object": w.name, "radius": 0.30, "axis": [0, 0, 1]} for w in road_wheels],
        "steering": {"object": f"{name}_wheel", "axis": [0, 0, 1], "ratio": 14.0},
    }
