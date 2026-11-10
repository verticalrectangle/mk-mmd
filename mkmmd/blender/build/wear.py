"""wear: a prop worn by a standing character: it rides a bone at the place its card says, with a strap fitted to her body and
its cable hung from the jack to the floor (docs/design.md: Prop card `use.wear`, Building: props).

[[prop]] wear = "rin"                          the prop's `use.wear` entry (the only one), worn by the cast member `rin`
[[prop]] wear = {cast = "rin", use = "stand", neck_deg = 20, yaw_deg = 5, roll_deg = 0, at = [x, y, z], pivot = [x, y, z],
                 scale = ["shoulder_width", "top", "top"],
                 cable = {trail = [dx, dy] (horizontal direction the cord lies away along on the floor), reach, tail_len,
                          slack, sway}}

The pose stage calls `apply` for every cast member (before the arms are solved, so hands can grip the worn prop) and
`fit_cables` once the base pose exists. The placement is mkmmd.core.wear (the card's numbers scaled by the wearer's
measure, `m.rig["measure"]`); the prop root is bone-parented with a transform relative to the bone's head frame. `strap`
(card entry): a band `<prop>_strap` from the card's `top` anchor over the wearer's shoulder (`over`, a semantic bone: its
joint) round her back to the `bottom` anchor, `width` x `thickness` metres, in the card's `material`: pulled taut over
her own surface as she stands at rest (`wearer_surface`, mkmmd.core.wear.strap_path), a standoff off it; how she
carries it (`_carry`: which points lie on her, hang free or belong to the prop, and the skin under them) is kept on the
prop (`worn["strap"]`) for the sim stage, which makes the band move (mkmmd.blender.build.sim.simulate_straps).
`cable`: the object `<prop>_cable` (a bevelled curve made by the prop's builder) is rebuilt from the card's `anchor` (the jack:
its empty and `dir`) to the floor under the wearer (the cast root's height) with mkmmd.core.cable.hang, unparented so
that its lower end stays put, and hooked at the jack with a smooth falloff so its top follows the prop."""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from ...core import cable as CB
from ...core import families as FAM
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


def wearer_surface(m):
    """(surface, skin) of the member for a strap. `surface(point, direction, far)` (mkmmd.core.wear.strap_path's): None,
    or (distance, facing, face) where a ray in the armature's frame first meets her (facing False: from within): her
    meshes as modelled (at rest), without the faces that belong to what hangs or swings (the bones of her rig's chains:
    hair, a skirt), to her arms from the upper arm down or to her head (by each vertex's heaviest bone): a strap lies
    on her trunk and her clothes, under her hair and clear of her chin. `skin(face)`: {bone: weight} with which that
    face moves (the mean of its corners' weights, summing to 1)."""
    arm = m.arm
    smap = S.semantic_map(arm)
    bones = arm.data.bones
    off = {b for c in (m.rig or {}).get("chains", []) for b in c["bones"]}
    for sem in ("arm.L", "arm.R", "head"):
        if sem in smap:
            off.add(smap[sem])
            off.update(b.name for b in bones[smap[sem]].children_recursive)
    inv = np.array(arm.matrix_world.inverted())
    verts, polys, owners, n0 = [], [], [], 0
    for o in S.model_meshes(arm):
        names = [g.name for g in o.vertex_groups]
        skip = np.zeros(len(o.data.vertices), bool)
        for i, v in enumerate(o.data.vertices):
            gs = [g for g in v.groups if g.group < len(names) and names[g.group] in bones]
            if gs:
                skip[i] = names[max(gs, key=lambda g: g.weight).group] in off
        V = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", V)
        M = inv @ np.array(o.matrix_world)
        verts.append(V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        for p in o.data.polygons:
            if not skip[list(p.vertices)].all():
                polys.append([i + n0 for i in p.vertices])
                owners.append((o, names, tuple(p.vertices)))
        n0 += len(o.data.vertices)
    if not polys:
        raise BuildError(f"{arm.name}: no surface to lay a strap on (no mesh faces on her trunk)")
    tree = BVHTree.FromPolygons(np.concatenate(verts).tolist(), polys)

    def surface(point, direction, far):
        d = Vector(direction)
        hit = tree.ray_cast(Vector(point), d, far)
        return None if hit[0] is None else (hit[3], hit[1].dot(d) < 0.0, hit[2])

    def skin(face):
        o, names, corners = owners[face]
        w = {}
        for i in corners:
            for g in o.data.vertices[i].groups:
                if g.group < len(names) and names[g.group] in bones:
                    w[names[g.group]] = w.get(names[g.group], 0.0) + g.weight
        total = sum(w.values())
        return {b: v / total for b, v in w.items()} if total > 0.0 else {}
    return surface, skin


def _carry(path, normals, kind, surface, skin, bone, width):
    """How the wearer carries a strap laid on her (`path`, `normals`, `kind` from mkmmd.core.wear.strap_path): kind (n,)
    of each point for the sim (0 held by the prop; 1 carried by her: from where the band first lies on her to where it
    last does; 2 free: the runs between her and a button), under (n, 3) the level of her surface under it (the highest
    of her under the band's `width` there, as a point under the band's line; NaN: none within 30 cm), and the bones
    and weights (n, len(bones)) with which that skin moves: smoothed along the band, a point with nothing under it
    taking its neighbours' (`bone` when she is under none of it)."""
    n = len(path)
    under, w = np.full((n, 3), np.nan), [None] * n
    across = np.cross(np.gradient(path, axis=0), normals)
    across /= np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-9)
    for i in range(n):
        met = [surface(path[i] + b * across[i] + 0.03 * normals[i], -normals[i], 0.33) for b in np.linspace(-0.42, 0.42, 7) * width]
        met = min((h for h in met if h is not None and h[1]), key=lambda h: h[0], default=None)
        if met is not None:
            under[i] = path[i] + (0.03 - met[0]) * normals[i]
            w[i] = skin(met[2]) or None
    kind = np.array(kind)
    on = np.flatnonzero(kind == 1)
    if len(on):
        kind[on[0]:on[-1] + 1] = 1
    has = np.array([i for i in range(n) if w[i]], int)
    if not len(has):
        return kind, under, [bone], np.ones((n, 1))
    bones = sorted({b for i in has for b in w[i]})
    W = np.zeros((n, len(bones)))
    for j, b in enumerate(bones):
        W[:, j] = np.interp(np.arange(n), has, [w[i].get(b, 0.0) for i in has])
    for _ in range(2):
        W[1:-1] = 0.25 * W[:-2] + 0.5 * W[1:-1] + 0.25 * W[2:]
    return kind, under, bones, W / W.sum(1, keepdims=True)


def _rings(path, normals, width, thick, r=0.0016, widths=None):
    """Rings (n, m, 3) of a rounded-rectangle band section (across x thickness) swept along `path` with the band's face
    along `normals`; `widths` (n,) narrows the band along the way (default: `width` all along)."""
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
    across = np.ones(len(P)) if widths is None else np.asarray(widths, float) / width
    return P[:, None, :] + (across[:, None, None] * sec[None, :, 0:1]) * B[:, None, :] \
        + (sec[None, :, 1:2] + thick / 2.0) * N[:, None, :]


STRAP_TAB = (0.6, 0.05)       # a strap's leather ends: this share of the band's width, reached over this many m from a button
STRAP_SLACK = 0.006           # a strap's free runs are this share longer than the straight way: they hang, a bar does not


def _tabs(path, normals, width):
    """A band with a strap's ends: (path, normals, widths) with the band narrowing to its tab's width (STRAP_TAB) toward
    each button and going a rounded tip past it, half the tab's width long (a band cut square at the button is a ribbon's
    end, and stands off the button as a flap)."""
    share, taper = STRAP_TAB
    P, N = np.asarray(path, float), np.asarray(normals, float)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    w = np.full(len(P), float(width))
    for along in (d, d[-1] - d):
        x = np.clip(along / taper, 0.0, 1.0)
        w = np.minimum(w, width * (share + (1.0 - share) * x * x * (3.0 - 2.0 * x)))
    rho = 0.5 * share * width
    turn = np.radians([8.0, 22.0, 40.0, 60.0, 80.0])             # the tip's half circle, from its end to the button

    def tip(p, q):
        u = (p - q) / max(float(np.linalg.norm(p - q)), 1e-9)
        return p + rho * np.cos(turn)[:, None] * u

    return (np.vstack([tip(P[0], P[1]), P, tip(P[-1], P[-2])[::-1]]),
            np.vstack([np.repeat(N[:1], len(turn), 0), N, np.repeat(N[-1:], len(turn), 0)]),
            np.r_[2.0 * rho * np.sin(turn), w, 2.0 * rho * np.sin(turn)[::-1]])


def band_mesh(path, normals, width, thick):
    """The strap's band as a mesh: its section swept along `path` (the inner face's line) facing along `normals`, with a
    tab at each end (`_tabs`). The same points in the same order for any path of as many points."""
    band, faces, widths = _tabs(path, normals, width)
    return SHL.cage(_rings(band, faces, width, thick, widths=widths), closed=True, caps=("fan", "fan"))


STRAP_COLLIDER = 0.04         # m: a strap's capsules for the strand sim are about this long


def build_strap(ctx, m, prop, sp, T, bone):
    """(report, state): the strap object of a worn prop (see the module doc) and what the sim stage needs to let it move
    (`simulate_straps`). `sp` is the entry's `strap` table with the project's overrides; `T` is the prop root's matrix in
    armature space and `bone` the bone it rides on."""
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
    radius = torso_radius(m.rig, smap.get("upper_body2", smap.get("upper_body")), (m.rig or {}).get("measure") or {})
    width, thick = float(sp.get("width", 0.05)), float(sp.get("thickness", 0.004))
    surface, skin = wearer_surface(m)
    standoff = float(sp.get("standoff", 0.007))
    path, normals, kind = WR.strap_path(ends[0], ends[1], shoulder, spine, radius, surface,
                                        over_side=math.copysign(1.0, shoulder[0]), standoff=standoff, width=width)
    mesh = band_mesh((path - o) @ R, normals @ R, width, thick)  # armature space -> the prop's frame (R is orthonormal)
    mat = bpy.data.materials.get(sp.get("material", ""))
    if mat is None:
        ctx.log("WARNING", f"prop {prop.name!r}: strap material {sp.get('material')!r} does not exist: the strap has none")
    name = f"{prop.name}_strap"
    obj = SH.mesh_object(name, mesh, collection("Props"), prop.root, lambda role: mat)
    # hair lies over a strap, not through it: capsules as wide as the band along it, their outside on its outer face,
    # with the prop's colliders (the strand sim's and the checks'), for the hair alone: clothes are under a strap
    out = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    line = (path - out * (0.5 * width - 0.5 * thick) - o) @ R
    seg = float(np.linalg.norm(np.diff(path, axis=0), axis=1).mean())
    at = list(range(0, len(line) - 1, max(1, int(round(STRAP_COLLIDER / max(seg, 1e-6)))))) + [len(line) - 1]
    caps = [{"type": "capsule", "object": obj.name, "a": [float(v) for v in line[i]], "b": [float(v) for v in line[j]],
             "R": 0.5 * width, "tag": f"{prop.name}:strap", "families": sorted(FAM.HAIR_FAMILIES)}
            for i, j in zip(at[:-1], at[1:])]
    prop.card = {**prop.card, "colliders": [*prop.card.get("colliders", []), *caps]}
    kind, under, bones, weights = _carry(path, normals, kind, surface, skin, bone, width)
    state = {"object": obj.name, "path": path, "normals": normals, "kind": kind, "under": under, "bones": bones,
             "weights": weights, "width": width, "thick": thick, "standoff": standoff,
             "slack": float(sp.get("slack", STRAP_SLACK)), "sim": bool(sp.get("sim", True))}
    return {"object": obj.name, "points": int(len(path)), "length_m": round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 3),
            "radius_m": round(radius, 4), "colliders": len(caps), "carried": int((kind == 1).sum()),
            "free": int((kind == 2).sum())}, state


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
        worn = {"cast": name, "entry": e, "T": T, "override": override, "cable": cable}
        if strap:
            rep["strap"], worn["strap"] = build_strap(ctx, m, prop, strap, T, bone)
        prop.worn = worn
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
        worn["floor_z"] = floor_z                                # the sim's floor: the wearer's root may be keyed off it
        radius = float(cs.get("radius", 0.0032))
        try:
            pts = CB.hang(top, d, floor_z, tail=trail, radius=radius,
                          **{k: float(cs[k]) for k in ("out", "sway", "reach", "tail_len", "slack") if k in cs})
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
