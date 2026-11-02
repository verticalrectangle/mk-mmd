"""The semi truck's card (pure Python, no bpy: the tests build it): what the builder (semi_truck.py) returns besides the
objects, assembled from the layout (semi_truck_layout), the options and the part modules' LOOKS and ANCHORS.

    wheel_keys(trailer)            the moving road-wheel parts, front to back (`<name>_<key>` objects)
    light_names(name, beams)       the real lights the builder makes: the headlamp beams, the dome light, the dash glow
    collider_boxes()               the hidden boxes for the sitters' hair and clothes: (base, centre, size, euler deg)
    card(name, roles, o, mods, lo, hi)
"""
import math

from . import semi_truck_layout as L

SIT = {"pelvis_deg": 8.0, "back_deg": L.BACK_RECLINE_DEG}
HIP_AIM = 0.03                    # the pose stage tips the pelvis after placing the hips: aim a little behind
TRAILER_PIVOT = (0.0, L.KINGPIN_Y, L.FIFTH_TOP_Z)   # the empty `<name>_trailer`: the van turns about the kingpin's axis


def _lst(v, nd=5):
    return [round(float(x), nd) for x in v]


def wheel_keys(trailer=True):
    """The dynamic part keys of every road wheel (a dual pair is one), front to back, L before R."""
    return [L.wheel_key(k, s) for k, _, kind, _ in L.axles() for s in ("L", "R") if trailer or kind != "trailer"]


def light_names(name, beams=True):
    """The names of the light objects the builder makes, in order."""
    return ([f"{name}_beam_L", f"{name}_beam_R"] if beams else []) + [f"{name}_dome", f"{name}_dash_glow"]


def collider_boxes():
    """[(base, centre, size, euler deg)]: the seats' cushions and backs, the doors, the floor, the dash and the windshield,
    in the prop's frame (the objects are `<name>_col_<base>`)."""
    out = []
    a = math.radians(L.BACK_RECLINE_DEG)
    up_back = (0.0, math.sin(a), math.cos(a))
    out_back = (0.0, -math.cos(a), math.sin(a))
    cy = 0.5 * (L.CUSHION_Y[0] + L.CUSHION_Y[1])
    for side, sx in (("driver", L.SEAT_X), ("passenger", -L.SEAT_X)):
        out.append((f"cushion_{side}", (sx, cy, L.CUSHION_TOP - 0.06), (0.52, L.CUSHION_Y[1] - L.CUSHION_Y[0], 0.12),
                    (0.0, 0.0, 0.0)))
        base = (sx, L.CUSHION_Y[1], L.CUSHION_TOP)
        c = tuple(b + u * 0.36 - n * 0.06 for b, u, n in zip(base, up_back, out_back))
        out.append((f"back_{side}", c, (0.52, 0.12, 0.72), (-L.BACK_RECLINE_DEG, 0.0, 0.0)))
    inner = L.CAB_HALF_W - L.CAB_WALL_T
    sill = L.DOOR_GLASS[0][1]
    for side, sx in (("L", 1.0), ("R", -1.0)):
        out.append((f"door_{side}", (sx * (inner + 0.02), 0.5 * (L.DOOR_Y[0] + L.DOOR_Y[1]), 0.5 * (L.CAB_FLOOR_Z + sill)),
                    (0.04, L.DOOR_Y[1] - L.DOOR_Y[0], sill - L.CAB_FLOOR_Z), (0.0, 0.0, 0.0)))
    out.append(("floor", (0.0, 0.5 * (L.DASH_Y[1] + L.BOC_Y), L.CAB_FLOOR_Z - 0.015),
                (2 * inner, L.BOC_Y - L.DASH_Y[1] + 0.3, 0.03), (0.0, 0.0, 0.0)))
    knee = L.CAB_FLOOR_Z + 0.30                                           # the dash's lowest edge over the driver's knees
    out.append(("dash", (0.0, 0.5 * (L.DASH_Y[0] + L.DASH_Y[1]), 0.5 * (L.DASH_TOP_Z + knee)),
                (2 * inner, L.DASH_Y[1] - L.DASH_Y[0], L.DASH_TOP_Z - knee), (0.0, 0.0, 0.0)))
    (y0, z0), (y1, z1) = L.WS_BASE, L.WS_TOP
    out.append(("windshield", (0.0, 0.5 * (y0 + y1), 0.5 * (z0 + z1)), (2 * L.WS_HALF_W[0], 0.03, math.hypot(y1 - y0, z1 - z0)),
                (L.WS_RAKE_DEG, 0.0, 0.0)))
    return out


def card(name, roles, o, mods, lo, hi):
    """The card of a truck named `name`: `roles` the colour roles as resolved ({role: spec}, the Pal's), `o` the options
    (semi_truck_layout.options), `mods` the part modules (their LOOKS and ANCHORS are merged in), `lo` / `hi` the bounds of
    the built geometry in the prop's frame."""
    sits, feet = [], []
    for side, sx in (("driver", 1.0), ("passenger", -1.0)):
        hx, hy, hz = L.HIP
        sits.append({"name": side, "hip": [sx * hx, hy + HIP_AIM, hz], "facing": [0, -1, 0], "seat_z": L.CUSHION_TOP,
                     "floor_z": L.CAB_FLOOR_Z, **SIT})
        fl, fr = L.FEET["L"], L.FEET["R"]
        if sx < 0:                                     # the passenger's feet: the driver's mirrored (L and R swap sides)
            fl, fr = (-fr[0], fr[1]), (-fl[0], fl[1])
        feet.append({"name": side, "L": [round(fl[0], 4), round(fl[1], 4), None], "R": [round(fr[0], 4), round(fr[1], 4), None],
                     "floor_z": L.CAB_FLOOR_Z})
    sill_z = L.DOOR_GLASS[0][1] - 0.02
    sill_x = L.CAB_HALF_W - L.CAB_WALL_T / 2
    y0, y1 = L.DOOR_GLASS[0][0] + 0.05, L.DOOR_GLASS[1][0] - 0.05
    rest = [{"name": f"sill_{s}", "type": "edge", "a": [x, y0, sill_z], "b": [x, y1, sill_z], "normal": [0, 0, 1]}
            for s, x in (("L", sill_x), ("R", -sill_x))]
    grip = [{"name": "wheel", "type": "ring", "center": _lst(L.WHEEL_C), "axis": _lst(L.WHEEL_AXIS), "radius": L.WHEEL_R,
             "tube": L.WHEEL_TUBE, "object": f"{name}_wheel"}]
    hx, hy, hz = L.HEADLAMP
    looks = {"road": (0.0, L.Y_BUMPER - 40.0, 1.8), "dash": L.GAUGES_C, "grille": (0.0, L.GRILLE_Y, 0.5 * sum(L.GRILLE_Z)),
             "headlamp_L": (hx, hy, hz), "headlamp_R": (-hx, hy, hz), "mirror_L": L.MIRROR,
             "mirror_R": (-L.MIRROR[0], L.MIRROR[1], L.MIRROR[2]), "fifth_wheel": (0.0, L.KINGPIN_Y, L.FIFTH_TOP_Z),
             "stack_L": (L.STACK_X, L.STACK_Y, L.STACK_TOP_Z), "stack_R": (-L.STACK_X, L.STACK_Y, L.STACK_TOP_Z),
             "driver": (L.SEAT_X, L.HIP[1], L.HIP[2] + 0.55)}
    for key, p in L.wheel_hubs().items():
        if o["trailer"] or not L.is_trailer(key):
            looks[key] = p
    if o["trailer"]:
        looks["tail"] = (0.0, L.TRAILER_Y[1], 2.0)
    anchors = {}
    for mod in mods.values():
        looks.update(getattr(mod, "LOOKS", {}))
        anchors.update(getattr(mod, "ANCHORS", {}))
    out = {
        "size": _lst([h - lw for lw, h in zip(lo, hi)], 3), "bounds": {"min": _lst(lo, 3), "max": _lst(hi, 3)},
        "origin": "floor_center", "front": "-Y", "blocks": True,
        "slots": {**roles, "trailer": o["trailer"], "beams": o["beams"], "mural_fit": o["mural_fit"],
                  **{f"mural_{k}": v for k, v in o["murals"].items()}},
        "use": {"sit": sits, "feet": feet, "grip": grip, "rest": rest,
                "look": [{"name": k, "point": _lst(v)} for k, v in looks.items()],
                "surface": L.surfaces(o["trailer"]),
                "anchor": [{"name": k, "point": _lst(a["point"]), **({"dir": _lst(a["dir"])} if "dir" in a else {})}
                           for k, a in anchors.items()]},
        "colliders": [{"type": "box", "object": f"{name}_col_{base}", "rnd": 0.01, "tag": name}
                      for base, *_ in collider_boxes()]
        + [{"type": "ring", "object": f"{name}_wheel", "radius": L.WHEEL_R, "tube": L.WHEEL_TUBE, "tag": name}],
        "wheels": [{"object": f"{name}_{k}", "radius": L.TYRE_R, "axis": [1.0, 0.0, 0.0]} for k in wheel_keys(o["trailer"])],
        "steering": {"object": f"{name}_wheel", "axis": [0.0, 0.0, 1.0], "ratio": L.STEER_RATIO, "wheelbase": L.WHEELBASE},
        "lights": light_names(name, o["beams"]), "params": {k: v[0] for k, v in L.PARAMS.items()},
    }
    if o["trailer"]:
        out["trailer"] = {"object": f"{name}_trailer", "pivot": _lst(TRAILER_PIVOT)}
    return out
