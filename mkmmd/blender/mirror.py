"""A window pane that shows what stands in front of it (docs/design.md: Shots, `reflection`): a plane light probe on the
glass object and a mirror layer in front of the glass shader. Screen-space ray tracing cannot see what is behind the
camera; EEVEE Next renders a plane probe's mirrored view of the whole scene, so a character who sits behind the camera
shows up in the pane.

    plane = glass_plane(glass)                     # centre, normal, in-plane axes and half extents in the world
    probe = find_probe(scene, plane) or make_probe(scene, "name", plane, {})
    undo = mirror_layer(glass, spec)               # spec: mkmmd.core.shotstyle.normalize_reflection(...)
    ...render...
    undo()                                         # the glass shader as it was; remove_probe(probe) for a made probe

Everything here changes the open scene only; callers restore it (mkmmd.blender.styles does, after every frame range)."""
import bpy
from mathutils import Matrix, Vector

from ..core import shotstyle as SS


def glass_plane(glass):
    """The glass object as a plane in the world: {centre, normal, axes (right, up unit vectors), half (2 extents)}. The
    normal is the first face's, so it points at the side the glass is seen from (a mesh facing the room)."""
    me = glass.data
    if not me.polygons:
        raise RuntimeError(f"reflection object {glass.name!r} has no faces")
    M = glass.matrix_world
    n = (M.to_3x3() @ me.polygons[0].normal).normalized()
    pts = [M @ v.co for v in me.vertices]
    c0 = sum(pts, Vector()) / len(pts)
    off = max(abs((p - c0).dot(n)) for p in pts)
    if off > 0.01:
        raise RuntimeError(f"reflection object {glass.name!r} is not flat (vertices up to {off * 1000:.0f} mm off the plane)")
    up = Vector((0, 0, 1)) - n * n.z
    if up.length < 1e-6:
        up = Vector((0, 1, 0)) - n * n.y
    up.normalize()
    right = up.cross(n).normalized()
    xs = [(p - c0).dot(right) for p in pts]
    ys = [(p - c0).dot(up) for p in pts]
    centre = c0 + right * ((max(xs) + min(xs)) / 2) + up * ((max(ys) + min(ys)) / 2)
    return {"centre": centre, "normal": n, "axes": (right, up), "half": ((max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2)}


def find_probe(sc, plane, tol=0.02):
    """A plane probe already standing on the glass (a set may carry one): same normal, within `tol` metres of it."""
    for o in sc.objects:
        if o.type == "LIGHT_PROBE" and o.data.type == "PLANE" and not o.hide_render:
            n = (o.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
            gap = abs((o.matrix_world.translation - plane["centre"]).dot(plane["normal"]))
            if n.dot(plane["normal"]) > 0.99 and gap < tol:
                return o
    return None


def make_probe(sc, name, plane, opts):
    """A plane light probe on the plane: local X / Y span the glass (`pad` times wider), local Z is the normal; it sits
    4 mm in front of the glass so the glass is behind its clip plane."""
    right, up = plane["axes"]
    hx, hy = plane["half"]
    pad = float(opts.get("pad", 1.02))
    pd = bpy.data.lightprobes.new(name, "PLANE")
    pd.influence_distance = float(opts.get("influence", 0.1))
    pd.clip_start = float(opts.get("clip", 0.001))
    ob = bpy.data.objects.new(name, pd)
    sc.collection.objects.link(ob)
    n = plane["normal"]
    c = plane["centre"] + n * 0.004
    ob.matrix_world = Matrix(((right.x * hx * pad, up.x * hy * pad, n.x, c.x),
                              (right.y * hx * pad, up.y * hy * pad, n.y, c.y),
                              (right.z * hx * pad, up.z * hy * pad, n.z, c.z),
                              (0, 0, 0, 1)))
    bpy.context.view_layer.update()
    return ob


def remove_probe(ob):
    pd = ob.data
    bpy.data.objects.remove(ob)
    if pd.users == 0:
        bpy.data.lightprobes.remove(pd)


def _output_node(nt):
    outs = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.target in ("ALL", "EEVEE")]
    return next((n for n in outs if n.is_active_output), outs[0] if outs else None)


def mirror_layer(glass, spec):
    """Mix a mirror into the glass material in front of its shader and return the function that takes it out again:
    Mix(fac = strength * (1 - backfacing), Mix(fac = dim, glass, black), Glossy(roughness, tint, the glass's normal map)).
    `dim` darkens the glass shader a little where the mirror shows, as glass does against a bright street."""
    mat = next((s.material for s in glass.material_slots if s.material is not None and s.material.use_nodes), None)
    if mat is None:
        raise RuntimeError(f"reflection object {glass.name!r} has no node material")
    nt = mat.node_tree
    out = _output_node(nt)
    if out is None or not out.inputs["Surface"].links:
        raise RuntimeError(f"material {mat.name!r}: no connected Material Output to put a mirror in front of")
    src = out.inputs["Surface"].links[0].from_socket
    new = []

    def node(kind, **props):
        n = nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        new.append(n)
        return n
    glossy = node("ShaderNodeBsdfGlossy")
    glossy.inputs["Roughness"].default_value = spec["roughness"]
    tint = spec.get("tint") or [1.0, 1.0, 1.0]
    glossy.inputs["Color"].default_value = (*(SS.srgb_to_linear(c) for c in tint), 1.0)
    nmap = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeNormalMap" and n not in new), None)
    if nmap is not None and spec.get("bend"):                # the rain on the glass bends the reflection too
        nt.links.new(nmap.outputs["Normal"], glossy.inputs["Normal"])
    black = node("ShaderNodeEmission")
    black.inputs["Strength"].default_value = 0.0
    dimmed = node("ShaderNodeMixShader")
    dimmed.inputs[0].default_value = spec["dim"]
    nt.links.new(src, dimmed.inputs[1])
    nt.links.new(black.outputs[0], dimmed.inputs[2])
    geo = node("ShaderNodeNewGeometry")
    front = node("ShaderNodeMath", operation="MULTIPLY_ADD")          # strength * (1 - backfacing) = -s * b + s
    front.inputs[1].default_value = -spec["strength"]
    front.inputs[2].default_value = spec["strength"]
    nt.links.new(geo.outputs["Backfacing"], front.inputs[0])
    mix = node("ShaderNodeMixShader")
    nt.links.new(front.outputs[0], mix.inputs[0])
    nt.links.new(dimmed.outputs[0], mix.inputs[1])
    nt.links.new(glossy.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])

    def undo():
        for n in new:
            nt.nodes.remove(n)
        nt.links.new(src, out.inputs["Surface"])
    return undo


def world_dark_to_camera(world):
    """Make the world black for camera rays (a plane probe captures with camera rays) while diffuse lighting keeps it:
    Mix(world shader, black, Is Camera Ray). Returns the function that undoes it, None for a world without nodes."""
    if world is None or not world.use_nodes or world.node_tree is None:
        return None
    nt = world.node_tree
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld" and n.is_active_output), None)
    if out is None or not out.inputs["Surface"].links:
        return None
    src = out.inputs["Surface"].links[0].from_socket
    black = nt.nodes.new("ShaderNodeBackground")
    black.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    black.inputs["Strength"].default_value = 0.0
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(src, mix.inputs[1])
    nt.links.new(black.outputs[0], mix.inputs[2])
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])

    def undo():
        for n in (black, path, mix):
            nt.nodes.remove(n)
        nt.links.new(src, out.inputs["Surface"])
    return undo
