"""wear: a prop worn by a standing character: it rides a bone at the place its card says, with a strap fitted to her body and
its cable hung from the jack to the floor (docs/design.md: Prop card `use.wear`, Building: props).

[[prop]] wear = "rin"                          the prop's `use.wear` entry (the only one), worn by the cast member `rin`
[[prop]] wear = {cast = "rin", use = "stand", neck_deg = 20, yaw_deg = 5, roll_deg = 0, at = [x, y, z], pivot = [x, y, z],
                 scale = ["shoulder_width", "top", "top"],
                 cable = {trail = [dx, dy] (horizontal direction the cord lies away along on the floor), reach, tail_len, sway}}

The pose stage calls `apply` for every cast member (before the arms are solved, so hands can grip the worn prop) and
`fit_cables` once the base pose exists. The placement is mkmmd.core.wear (the card's numbers scaled by the wearer's
measure, `m.rig["measure"]`); the prop root is bone-parented with a transform relative to the bone's head frame. `strap`
(card entry): a band `<prop>_strap` from the card's `top` anchor over the wearer's shoulder (`over`, a semantic bone: its
joint) round her back to the `bottom` anchor, `width` x `thickness` metres, in the card's `material`: its centre line
follows her torso at the radius of her chest's collision body (rig.json) plus a standoff (mkmmd.core.wear.strap_path).
`cable`: the object `<prop>_cable` (a bevelled curve made by the prop's builder) is rebuilt from the card's `anchor` (the jack:
its empty and `dir`) to the floor under the wearer (the cast root's height) with mkmmd.core.cable.hang, unparented so
that its lower end stays put, and hooked at the jack with a smooth falloff so its top follows the prop."""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ...core import cable as CB
from ...core import shell as SHL
from ...core import wear as WR
from .. import keys as K
from .. import scene as S
from ..library import shell as SH
from . import BuildError, collection

TORSO_SHARE = 0.55         # torso half depth as a share of the shoulder width when the rig has no chest body


def spec_of(prop_spec):
    """(cast, use, override) of a [[prop]] `wear`, or None."""
    w = prop_spec.get("wear")
    if not w:
        return None
    if isinstance(w, str):
        return w, None, {}
    w = dict(w)
    try:
        cast = w.pop("cast")
    except KeyError:
        raise BuildError(f"prop {prop_spec['name']!r}: wear = {{...}} needs `cast` (who wears it)") from None
    return cast, w.pop("use", None), w


def torso_radius(rig, bone, measure):
    """Half depth of the torso at the chest: the chest bone's capsule body in rig.json, else a share of the shoulder width."""
    for b in (rig or {}).get("bodies", []):
        g = b.get("geom") or {}
        if b.get("bone") == bone and g.get("kind") == "capsule":
            return float(g["R"])
    return TORSO_SHARE * float(measure.get("shoulder_width", 0.4))


def _heads(arm, smap, names):
    return np.array([np.array(arm.data.bones[smap[n]].head_local) for n in names if n in smap])


def _rings(path, normals, width, thick, r=0.0016):
    """Rings (n, m, 3) of a rounded-rectangle band section (across x thickness) swept along `path` with the band's face
    along `normals`."""
    P = np.asarray(path, float)
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1), 1e-9)[:, None]
    N = np.asarray(normals, float)
    N = N - np.einsum("ij,ij->i", N, T)[:, None] * T
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-9)[:, None]
    for _ in range(3):                                           # smooth the band's roll along the path
        N[1:-1] = 0.25 * N[:-2] + 0.5 * N[1:-1] + 0.25 * N[2:]
        N -= np.einsum("ij,ij->i", N, T)[:, None] * T
        N /= np.maximum(np.linalg.norm(N, axis=1), 1e-9)[:, None]
    B = np.cross(T, N)
    sec = SHL.rrect(width, thick, r, 3)
    return P[:, None, :] + sec[None, :, 0:1] * B[:, None, :] + (sec[None, :, 1:2] + thick / 2.0) * N[:, None, :]


def build_strap(ctx, m, prop, sp, T):
    """The strap object of a worn prop (see the module doc): `sp` is the entry's `strap` table with the project's overrides;
    `T` is the prop root's matrix in armature space."""
    arm = m.arm
    smap = S.semantic_map(arm)
    anchors = {a["name"]: a for a in prop.card.get("use", {}).get("anchor", [])}
    for key in ("top", "bottom"):
        if sp.get(key) not in anchors:
            raise BuildError(f"prop {prop.name!r}: wear strap {key} = {sp.get(key)!r} is not one of its anchors "
                             f"({sorted(anchors)})")
    R, o = T[:3, :3], T[:3, 3]
    ends = []
    for key in ("top", "bottom"):
        a = anchors[sp[key]]
        ends.append((R @ np.asarray(a["point"], float) + o, R @ np.asarray(a.get("dir", (0.0, 0.0, 1.0)), float)))
    over = str(sp.get("over", "arm.L")).replace("shoulder.", "arm.")
    if over not in smap:
        raise BuildError(f"prop {prop.name!r}: the strap goes over {over!r}, which the model lacks")
    shoulder = np.array(arm.data.bones[smap[over]].head_local)
    spine = WR.spine_axis(_heads(arm, smap, ("lower_body", "upper_body", "upper_body2", "neck")))
    measure = (m.rig or {}).get("measure") or {}
    radius = torso_radius(m.rig, smap.get("upper_body2", smap.get("upper_body")), measure)
    path = WR.strap_path(ends[0], ends[1], shoulder, spine, radius, over_side=math.copysign(1.0, shoulder[0]),
                         shoulder_r=float(sp.get("shoulder_radius", 0.22 * float(measure.get("upper_arm", 0.2)))))
    c = np.array([spine(z)[1] for z in path[:, 2]])
    centre = np.stack([np.zeros(len(path)), c, path[:, 2] - np.where(path[:, 2] > shoulder[2] - 0.06, 0.07, 0.0)], 1)
    normals = path - centre
    rings = _rings(path, normals, float(sp.get("width", 0.05)), float(sp.get("thickness", 0.004)))
    rings = (rings.reshape(-1, 3) - o) @ R                       # armature space -> the prop's frame (R is orthonormal)
    rings = rings.reshape(len(path), -1, 3)
    mesh = SHL.cage(rings, closed=True, caps=("fan", "fan"))
    mat = bpy.data.materials.get(sp.get("material", ""))
    if mat is None:
        ctx.log("WARNING", f"prop {prop.name!r}: strap material {sp.get('material')!r} does not exist: the strap has none")
    name = f"{prop.name}_strap"
    obj = SH.mesh_object(name, mesh, collection("Props"), prop.root, lambda role: mat)
    return {"object": obj.name, "points": int(len(path)), "length_m": round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 3),
            "radius_m": round(radius, 4)}


def apply(ctx, name, m):
    """Wear every prop whose `wear` names the cast member `name`: bone-parent it at its card's placement and build its strap.
    Returns {prop: report}."""
    from . import pose as POSE
    out = {}
    for spec in ctx.data.get("prop", []):
        w = spec_of(spec)
        if not w or w[0] != name:
            continue
        _cast, use, override = w
        prop = ctx.props[spec["name"]]
        try:
            e = WR.entry(prop.card.get("use"), use)
            bone = S.resolve_bone(m.arm, e["bone"])
            measure = (m.rig or {}).get("measure")
            if not measure:
                raise WR.WearError(f"the model has no measurements: wear needs the rig.json of cast {name!r} (`measure`)")
            pb = m.arm.data.bones[bone]
            T = WR.matrix(e, measure, np.array(pb.head_local), override)
            strap, cable = WR.table(e, override, "strap"), WR.table(e, override, "cable")
        except (WR.WearError, KeyError) as ex:
            raise BuildError(f"prop {spec['name']!r}: wear: {ex}")
        POSE.attach_to_bone(prop.root, m.arm, bone, K.mat(np.linalg.inv(np.array(pb.matrix_local)) @ T))
        bpy.context.view_layer.update()
        rep = {"bone": bone, "entry": e["name"], "at_mm": [round(float(v) * 1e3) for v in T[:3, 3]]}
        if strap:
            rep["strap"] = build_strap(ctx, m, prop, strap, T)
        prop.worn = {"cast": name, "entry": e, "T": T, "override": override, "cable": cable}
        out[spec["name"]] = rep
        ctx.log("wear", spec["name"], name, bone)
    return out


def fit_cables(ctx):
    """Re-hang the cable of every worn prop from its jack to the floor under its wearer, at the settled base pose."""
    out = {}
    sc = bpy.context.scene
    keep = sc.frame_current
    for pname, prop in ctx.props.items():
        worn = getattr(prop, "worn", None)
        if not worn or not worn["cable"]:
            continue
        cs = worn["cable"]
        m = ctx.cast[worn["cast"]]
        anchors = {a["name"]: a for a in prop.card.get("use", {}).get("anchor", [])}
        a = anchors.get(cs.get("anchor", "jack"))
        obj = bpy.data.objects.get(cs.get("object", f"{pname}_cable"))
        if a is None or obj is None or not a.get("object") or a["object"] not in bpy.data.objects:
            raise BuildError(f"prop {pname!r}: wear cable needs an anchor with an `object` and the cable object "
                             f"({cs.get('object')!r}); the card's anchors: {sorted(anchors)}")
        sc.frame_set(ctx.start + ctx.settle)
        jack = bpy.data.objects[a["object"]]
        top = np.array(jack.matrix_world.translation)
        d = np.array((prop.root.matrix_world.to_3x3() @ Vector(a.get("dir", (0.0, 0.0, -1.0)))).normalized())
        Rw = m.arm.matrix_world.to_3x3()
        lat, back = np.array(Rw @ Vector((1, 0, 0))), np.array(Rw @ Vector((0, 1, 0)))
        trail = cs.get("trail")
        if trail is None:
            trail = (-lat + back)[:2]                           # behind and to her right: toward the amp she is plugged into
        floor_z = float(m.root.matrix_world.translation.z)
        radius = float(cs.get("radius", 0.0032))
        try:
            pts = CB.hang(top, d, floor_z, tail=trail, radius=radius,
                          **{k: float(cs[k]) for k in ("out", "sway", "reach", "tail_len") if k in cs})
        except ValueError as ex:
            raise BuildError(f"prop {pname!r}: wear cable: {ex}")
        cu = obj.data
        cu.splines.clear()
        sp = cu.splines.new("NURBS")
        sp.points.add(len(pts) - 1)
        sp.order_u = 4
        sp.use_endpoint_u = True
        for pt, p in zip(sp.points, pts):
            pt.co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
        cu.bevel_depth = radius
        for md in [md for md in obj.modifiers if md.type == "HOOK"]:
            obj.modifiers.remove(md)
        obj.parent = None                                        # the lower end stays where it lies
        obj.matrix_world = Matrix()
        bpy.context.view_layer.update()
        h = obj.modifiers.new("hook_jack", "HOOK")
        h.object = jack
        h.center = Vector(top)
        h.falloff_type = "SMOOTH"
        h.falloff_radius = float(cs.get("follow", 0.35))
        h.use_falloff_uniform = True
        h.matrix_inverse = jack.matrix_world.inverted()
        out[pname] = {"points": int(len(pts)), "top": [round(float(v), 3) for v in top], "floor_z": round(floor_z, 3),
                      "length_m": round(CB.length(pts), 2), "follow_m": h.falloff_radius}
        ctx.log("cable", pname, out[pname]["length_m"], "m")
    sc.frame_set(keep)
    return out
