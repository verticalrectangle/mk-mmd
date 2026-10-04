"""A neutral review studio for a single model: flat grey world, soft round floor, key + fill + rim suns (the same rig the
Rin study scenes use, so renders compare side by side with other models), Standard view transform."""
import math

import bpy
from mathutils import Vector

BACKDROP = 0.42                 # linear value of the flat grey world (about sRGB #6c6c6c under Standard)

LIGHTS = (                      # name, colour, power, position (m), spread angle (deg), shadows
    ("key", (1.0, 0.957, 0.902), 2.6, (2.5, -4.0, 4.0), 18.0, True),
    ("fill", (0.91, 0.94, 1.0), 1.1, (-3.5, -3.0, 1.5), 40.0, False),
    ("rim", (1.0, 1.0, 1.0), 0.9, (-1.5, 3.5, 3.0), 20.0, True),
)


def _world():
    w = bpy.data.worlds.new("studio")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (BACKDROP, BACKDROP, BACKDROP, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs[0])
    bpy.context.scene.world = w


def _floor(radius=4.0):
    import bmesh
    me = bpy.data.meshes.new("studio_floor")
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, cap_tris=False, segments=96, radius=radius)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("studio_floor", me)
    bpy.context.scene.collection.objects.link(ob)
    mat = bpy.data.materials.new("studio_floor")
    mat.use_nodes = True
    mat.surface_render_method = "BLENDED"
    nt = mat.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vm = nt.nodes.new("ShaderNodeVectorMath")
    vm.operation = "LENGTH"
    nt.links.new(tc.outputs["Object"], vm.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.22 * radius
    mr.inputs["From Max"].default_value = 0.9 * radius
    mr.inputs["To Min"].default_value = 1.0
    mr.inputs["To Max"].default_value = 0.0
    mr.clamp = True
    nt.links.new(vm.outputs["Value"], mr.inputs["Value"])
    sq = nt.nodes.new("ShaderNodeMath")
    sq.operation = "MULTIPLY"
    nt.links.new(mr.outputs[0], sq.inputs[0])
    nt.links.new(mr.outputs[0], sq.inputs[1])
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    diff.inputs["Color"].default_value = (BACKDROP, BACKDROP, BACKDROP, 1.0)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(sq.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(diff.outputs[0], mix.inputs[2])
    mo = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(mix.outputs[0], mo.inputs[0])
    ob.data.materials.append(mat)
    ob.visible_glossy = False
    return ob


def _sun(name, color, power, at, angle_deg, shadow, target):
    ld = bpy.data.lights.new(f"studio_{name}", "SUN")
    ld.color = color
    ld.energy = power
    ld.angle = math.radians(angle_deg)
    ld.use_shadow = shadow
    ob = bpy.data.objects.new(f"studio_{name}", ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = at
    ob.rotation_euler = (Vector(target) - Vector(at)).to_track_quat("-Z", "Y").to_euler()
    return ob


def setup(height=1.6):
    """Add the studio to the current scene. Returns the object names it created."""
    sc = bpy.context.scene
    _world()
    made = [_floor().name]
    target = (0.0, 0.0, 0.62 * height)
    for name, color, power, at, ang, shadow in LIGHTS:
        made.append(_sun(name, color, power, at, ang, shadow, target).name)
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.eevee.use_shadows = True
    sc.render.film_transparent = False
    return made
