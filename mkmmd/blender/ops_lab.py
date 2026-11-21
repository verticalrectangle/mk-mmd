"""lab_build: build one case of the parity lab (mkmmd/render/lab.py; docs/design.md: Parity), a small scene that
isolates one thing EEVEE does, render Blender's frames of it and bake it; lab_variant: render a built case's saved
scene again as a variant of it (shadows off, Cycles) without touching the case.

args: {"case": the case spec (mkmmd.render.lab: name, kind, frames, refs, outputs, render, params), "dir": the case's
folder (a tiny mk project whose mk.toml names scene.blend; the job's project is it), "armature": the cast member's
armature for the MMD kinds (the job then opens the member's built scene and strips it to that member)}.

The op builds the scene from the factory default (or the stripped source), saves <dir>/scene.blend, renders every
reference frame of every output once with EEVEE and writes it three ways from the same render result:
  ref/<output>/<frame>.png      8-bit RGB, as `mk render` writes a frame (view transform, dither)
  ref16/<output>/<frame>.png    16-bit RGB: the same picture before the 8-bit quantisation and its dither
  linear/<output>/<frame>.exr   scene-linear RGB (half float, no view transform): the film's filtered radiance
and then bakes the saved scene with the build's own bake stage (mkmmd.blender.build.bake.run, beside the .blend:
scene.bake/). A bake that fails is reported, not raised: the frames stay and `mk parity lab --rebake` bakes again.

lab_variant (args: "case", "variant", "out") opens the saved scene instead, changes it in memory (every light's shadow
off; Cycles with direct light only) and renders the same reference frames into out/<output>/ (8-bit PNG and the
scene-linear EXR beside it) and out/16/<output>/ (16-bit PNG); it saves nothing.

Kinds (each reads the params mkmmd.render.lab gives it): light (one light over a diffuse plane, a diffuse sphere and a
glossy one), shadow (objects on or over the plane, one light), spheres (one sphere per frame, each its own material),
ramp (emission strips through the view transform), edge (emission shapes for the film filter), alpha (see-through
planes over a checkerboard), bend (a skinned tube bending), blur (a moving sphere, a spinning bar, then a panning
camera), motion (things moving for the motion blur cases over a checkered wall: rows crossing the frame, turning props,
a skinned arm, a keyed camera; the case's shutter, its position and steps), dof (emission checkerboards at several
depths and bright dots through a camera with depth of field keyed per
frame), scene (objects in materials whose inputs may change per frame, lights, a camera keyed per frame: with the
case's world and ray tracing), mmd_model (a cast member at rest, one light per frame, then all of them), mmd_sphere (a
sphere in each of the member's most used materials). A case with a `world` gets one (world(): uniform, a sky with a
sun disc, stripes, a Light Path split between camera rays and the rest); the others have none."""
import json
import math
import os
import time

import bmesh
import bpy
from mathutils import Vector

from . import keys as K
from . import scene as S
from .runtime import CTX, op

SHOT = "lab"


# ================================================================================================= meshes
def _link(ob):
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _mesh(name, fill, mats, smooth=True, at=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    """A mesh object from `fill(bm, uv)` (a bmesh with a UVMap layer), its materials in slot order."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    fill(bm, uv)
    for f in bm.faces:
        f.smooth = smooth and len(f.verts) <= 4             # n-gon caps stay flat
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    ob = _link(bpy.data.objects.new(name, me))
    ob.location, ob.rotation_euler = at, rot
    return ob


def _quad(bm, uv, w, h, x0=0.0, y0=0.0, slot=0):
    """A w x h quad on the XY plane centred at (x0, y0), facing +Z, UV 0..1 across it."""
    vs = [bm.verts.new((x0 + x * w / 2, y0 + y * h / 2, 0.0)) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    f = bm.faces.new(vs)
    f.material_index = slot
    for loop, c in zip(f.loops, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
        loop[uv].uv = c
    return f


def plane(name, w, h, mat, at=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    return _mesh(name, lambda bm, uv: _quad(bm, uv, w, h), [mat], smooth=False, at=at, rot=rot)


def upright(name, w, h, mat, at, spin=0.0):
    """A w x h quad standing in the XZ plane facing -Y (toward a camera on -Y looking +Y), turned `spin` degrees in
    its own plane; UV u runs along +X, v up."""
    return plane(name, w, h, mat, at=at, rot=(math.pi / 2, math.radians(spin), 0.0))


def checker_fill(nx, nz, cell):
    """A bmesh fill of nx x nz quads of `cell` metres centred on the origin of the XY plane, alternating material slots
    0 and 1."""
    def fill(bm, uv):
        for i in range(nx):
            for k in range(nz):
                _quad(bm, uv, cell, cell, (i - nx / 2 + 0.5) * cell, (k - nz / 2 + 0.5) * cell, slot=(i + k) % 2)
    return fill


def sphere(name, r, at, mat, u=128, v=64):
    return _mesh(name, lambda bm, uv: bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=r, calc_uvs=True),
                 [mat], at=at)


def box(name, size, at, mat, yaw=0.0):
    """A box of `size` (x, y, z) centred at `at`, turned `yaw` degrees about Z, flat shaded."""
    def fill(bm, uv):
        bmesh.ops.create_cube(bm, size=1.0, calc_uvs=True)
        bmesh.ops.scale(bm, vec=tuple(size), verts=bm.verts)
    return _mesh(name, fill, [mat], smooth=False, at=at, rot=(0.0, 0.0, math.radians(yaw)))


def tube(name, r, h, mat, at, segments=48, rings=1):
    """A capped cylinder from z = 0 to z = h (smooth sides, flat caps), `rings` rows of quads up its side."""
    def fill(bm, uv):
        rows = [[bm.verts.new((r * math.cos(2 * math.pi * i / segments), r * math.sin(2 * math.pi * i / segments),
                               h * k / rings)) for i in range(segments)] for k in range(rings + 1)]
        for k in range(rings):
            for i in range(segments):
                j = (i + 1) % segments
                f = bm.faces.new((rows[k][i], rows[k][j], rows[k + 1][j], rows[k + 1][i]))
                for loop, c in zip(f.loops, ((i, k), (i + 1, k), (i + 1, k + 1), (i, k + 1))):
                    loop[uv].uv = (c[0] / segments, c[1] / rings)
        for ring in (list(reversed(rows[0])), rows[-1]):
            f = bm.faces.new(ring)
            for loop in f.loops:
                loop[uv].uv = (0.5 + 0.5 * loop.vert.co.x / r, 0.5 + 0.5 * loop.vert.co.y / r)
    return _mesh(name, fill, [mat], at=at)


# ================================================================================================= materials
def _set(sock, value):
    if sock.type == "RGBA" and len(value) == 3:
        value = (*value, 1.0)
    sock.default_value = value


def _material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    return m, nt, nt.nodes.new("ShaderNodeOutputMaterial")


def shader(name, kind, inputs=None, distribution=None):
    """A material of one BSDF node: diffuse, glossy (the Glossy BSDF), principled, emission, translucent."""
    idname = {"diffuse": "ShaderNodeBsdfDiffuse", "glossy": "ShaderNodeBsdfAnisotropic",
              "principled": "ShaderNodeBsdfPrincipled", "emission": "ShaderNodeEmission",
              "translucent": "ShaderNodeBsdfTranslucent"}[kind]
    m, nt, out = _material(name)
    n = nt.nodes.new(idname)
    if distribution:
        n.distribution = distribution
    for k, v in (inputs or {}).items():
        _set(n.inputs[k], v)
    nt.links.new(n.outputs[0], out.inputs["Surface"])
    return m


def ramp_material(name, color, lo, hi):
    """Emission colour x 2^(lo + (hi - lo) u), u the UV's x: a log ramp of scene-linear values across the face."""
    m, nt, out = _material(name)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ma = nt.nodes.new("ShaderNodeMath")
    ma.operation = "MULTIPLY_ADD"
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    nt.links.new(sep.outputs["X"], ma.inputs[0])
    ma.inputs[1].default_value, ma.inputs[2].default_value = hi - lo, lo
    pw.inputs[0].default_value = 2.0
    nt.links.new(ma.outputs[0], pw.inputs[1])
    _set(em.inputs["Color"], color)
    nt.links.new(pw.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


def alpha_material(name, color, alpha, method):
    """Emission mixed with a Transparent BSDF by `alpha` (a number, or "u": the UV's x), as MMD materials mix theirs;
    `method` the material's surface render method (DITHERED: hashed, BLENDED)."""
    m, nt, out = _material(name)
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    _set(em.inputs["Color"], color)
    if alpha == "u":
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["UV"], sep.inputs[0])
        nt.links.new(sep.outputs["X"], mix.inputs["Fac"])
    else:
        mix.inputs["Fac"].default_value = float(alpha)
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.surface_render_method = method
    return m


# ================================================================================================= lights, cameras, keys
def light(name, spec):
    """A light from a lab spec: type, energy, color, shape / size / size_y (area), radius (point, spot), spot (degrees),
    blend (spot), angle (degrees, sun), props (any other light property by identifier: shadow_filter_radius, ...).
    Shadows on, as mk's lights have them."""
    ld = bpy.data.lights.new(name, spec["type"])
    ld.energy = float(spec["energy"])
    ld.color = spec.get("color", (1.0, 1.0, 1.0))
    ld.use_shadow = True
    if spec["type"] == "AREA":
        ld.shape = spec.get("shape", "SQUARE")
        ld.size = float(spec.get("size", 1.0))
        ld.size_y = float(spec.get("size_y", spec.get("size", 1.0)))
    if spec["type"] in ("POINT", "SPOT"):
        ld.shadow_soft_size = float(spec.get("radius", 0.1))
    if spec["type"] == "SPOT":
        ld.spot_size = math.radians(float(spec.get("spot", 45.0)))
        ld.spot_blend = float(spec.get("blend", 0.15))
    if spec["type"] == "SUN":
        ld.angle = math.radians(float(spec.get("angle", 1.0)))
    for k, v in (spec.get("props") or {}).items():
        setattr(ld, k, v)
    return _link(bpy.data.objects.new(name, ld))


def aim_euler(at, target):
    return (Vector(target) - Vector(at)).to_track_quat("-Z", "Y").to_euler()


def place(ob, at, target):
    ob.location = at
    ob.rotation_euler = aim_euler(at, target)


def key_places(ob, frames, places, interp="CONSTANT"):
    """Key location and aim [(at, target)] on `frames`."""
    K.key_vec(ob, "location", frames, [p[0] for p in places], interp=interp)
    K.key_vec(ob, "rotation_euler", frames, [tuple(aim_euler(*p)) for p in places], interp=interp)


def show_on(obs_by_frame, frames):
    """Key hide_render so that on frames[i] only the objects of obs_by_frame[i] render (of all of them)."""
    every = {o.name: o for obs in obs_by_frame for o in obs}
    for ob in every.values():
        vals = [0.0 if any(o.name == ob.name for o in obs) else 1.0 for obs in obs_by_frame]
        K.key_prop(ob, "hide_render", frames, vals, interp="CONSTANT")


def camera(name, at, target, lens=50.0, ortho=None):
    cd = bpy.data.cameras.new(name)
    cd.lens, cd.sensor_width, cd.sensor_fit = float(lens), 36.0, "AUTO"
    cd.clip_start, cd.clip_end = 0.05, 200.0
    if ortho:
        cd.type, cd.ortho_scale = "ORTHO", float(ortho)
    ob = _link(bpy.data.objects.new(name, cd))
    place(ob, at, target)
    return ob


def cameras(outs, at, target, lens=45.0, ortho=None):
    """The shot's camera of every output, all in one place."""
    return {o: camera(f"{SHOT}@{o}", at, target, lens, ortho) for o in outs}


def floor(albedo=0.5, size=12.0):
    return plane("floor", size, size, shader("floor", "diffuse", {"Color": (albedo,) * 3, "Roughness": 0.0}))


# ================================================================================================= kinds
def build_light(p, frames, outs):
    floor()
    sphere("ball_diffuse", 0.5, (-0.7, 0.0, 0.5), shader("diffuse", "diffuse", {"Color": (0.8, 0.8, 0.8)}))
    sphere("ball_glossy", 0.5, (0.7, 0.0, 0.5), shader("glossy", "glossy", {"Color": (0.8, 0.8, 0.8),
                                                                            "Roughness": p.get("roughness", 0.25)}))
    key_places(light("light", p["light"]), frames, [tuple(pl) for pl in p["places"]])
    return {"cameras": cameras(outs, *p["camera"], lens=p.get("lens", 45.0))}


def see_through(name, color, alpha):
    """A diffuse material of `color` mixed with a Transparent BSDF by `alpha`, hashed (DITHERED): opaque when alpha is 1."""
    if alpha >= 1.0:
        return shader(name, "diffuse", {"Color": tuple(color)})
    m, nt, out = _material(name)
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    di = nt.nodes.new("ShaderNodeBsdfDiffuse")
    _set(di.inputs["Color"], tuple(color))
    mix.inputs["Fac"].default_value = float(alpha)
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(di.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "DITHERED"
    return m


def build_shadow(p, frames, outs):
    """Objects on or over the plane, one light keyed to `places`: boxes (size, yaw), spheres (r), tubes (r, h, standing
    on their place), plates (size [w, h]); `rot` (degrees about x, y, z) turns any of them; each grey diffuse unless it
    has a `color` or an `alpha` (hashed see-through), and `camera: false` hides it from the camera (it still casts).
    `cameras` ({output: [at, target, lens]}) places each output's camera on its own."""
    floor(p.get("floor", 0.6), p.get("floor_size", 12.0))
    grey = shader("grey", "diffuse", {"Color": (0.7, 0.7, 0.7)})
    for o in p["objects"]:
        mat = grey
        if "alpha" in o or "color" in o:
            mat = see_through(o["name"], o.get("color", (0.7, 0.7, 0.7)), float(o.get("alpha", 1.0)))
        if o["kind"] == "box":
            ob = box(o["name"], o["size"], o["at"], mat, o.get("yaw", 0.0))
        elif o["kind"] == "sphere":
            ob = sphere(o["name"], o["r"], o["at"], mat)
        elif o["kind"] == "plate":
            ob = plane(o["name"], o["size"][0], o["size"][1], mat, at=o["at"])
        else:
            ob = tube(o["name"], o["r"], o["h"], mat, o["at"])
        if "rot" in o:
            ob.rotation_euler = [math.radians(a) for a in o["rot"]]
        ob.visible_camera = bool(o.get("camera", True))
    ob = light("light", p["light"])
    key_places(ob, frames, [tuple(pl) for pl in p["places"]])
    for prop, values in (p.get("light_keys") or {}).items():
        K.key_prop(ob.data, prop, frames, values, interp="CONSTANT")
    if p.get("cameras"):
        return {"cameras": {o: camera(f"{SHOT}@{o}", *p["cameras"][o][:2], lens=(p["cameras"][o][2:] or [45.0])[0])
                            for o in outs}}
    return {"cameras": cameras(outs, *p["camera"], lens=p.get("lens", 45.0))}


def build_spheres(p, frames, outs):
    floor(p.get("floor", 0.05))
    r = p.get("r", 0.6)
    balls = [[sphere(f"ball_{i:02d}", r, (0.0, 0.0, r), shader(m["name"], m["kind"], m.get("inputs"),
                                                              m.get("distribution")))]
             for i, m in enumerate(p["materials"])]
    show_on(balls, frames)
    place(light("light", p["light"]), *p["place"])
    return {"cameras": cameras(outs, *p["camera"], lens=p.get("lens", 45.0))}


def build_ramp(p, frames, outs):
    rows, w, h = p["rows"], p["width"], p["height"]
    for i, c in enumerate(rows):
        z = h / 2 - (i + 0.5) * h / len(rows)
        upright(f"strip_{i}", w, h / len(rows), ramp_material(f"ramp_{i}", c, p["lo"], p["hi"]), (0.0, 0.0, z))
    return {"cameras": cameras(outs, (0.0, -10.0, 0.0), (0.0, 0.0, 0.0), ortho=w)}


def build_edge(p, frames, outs):
    for s in p["shapes"]:
        m = shader(f"em_{s['name']}", "emission", {"Color": s.get("color", (1.0, 1.0, 1.0)),
                                                   "Strength": s.get("strength", 1.0)})
        upright(s["name"], s["w"], s["h"], m, (s["at"][0], 0.0, s["at"][1]), spin=s.get("spin", 0.0))
    return {"cameras": cameras(outs, (0.0, -10.0, 0.0), (0.0, 0.0, 0.0), ortho=p["width"])}


def build_alpha(p, frames, outs):
    nx, nz, cell = p["checker"]
    light_m = shader("check_light", "emission", {"Color": (0.9, 0.9, 0.9)})
    dark_m = shader("check_dark", "emission", {"Color": (0.05, 0.05, 0.05)})
    _mesh("checker", checker_fill(nx, nz, cell), [light_m, dark_m], smooth=False, at=(0.0, 1.0, 0.0),
          rot=(math.pi / 2, 0.0, 0.0))
    for s in p["planes"]:
        upright(s["name"], s["w"], s["h"], alpha_material(s["name"], s["color"], s["alpha"], p["method"]),
                (s["at"][0], s["y"], s["at"][1]))
    return {"cameras": cameras(outs, (0.0, -12.0, 0.0), (0.0, 0.0, 0.0), ortho=p["width"])}


def build_bend(p, frames, outs):
    floor()
    arm_data = bpy.data.armatures.new("bend_arm")
    arm = _link(bpy.data.objects.new("bend_arm", arm_data))
    arm.location = p["at"]
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    lower = arm_data.edit_bones.new("lower")
    lower.head, lower.tail = (0.0, 0.0, 0.0), (0.0, 0.0, p["h"] / 2)
    upper = arm_data.edit_bones.new("upper")
    upper.head, upper.tail = (0.0, 0.0, p["h"] / 2), (0.0, 0.0, p["h"])
    upper.parent, upper.use_connect = lower, True
    bpy.ops.object.mode_set(mode="OBJECT")
    ob = tube("bend", p["r"], p["h"], shader("bend", "diffuse", {"Color": (0.75, 0.6, 0.45)}), (0.0, 0.0, 0.0),
              segments=48, rings=p["rings"])
    ob.parent = arm
    lo, up = ob.vertex_groups.new(name="lower"), ob.vertex_groups.new(name="upper")
    half, blend = p["h"] / 2, p["blend"]
    for v in ob.data.vertices:                     # smoothstep from the lower bone to the upper one round the joint
        t = min(1.0, max(0.0, (v.co.z - (half - blend)) / (2 * blend)))
        w = t * t * (3 - 2 * t)
        if w < 1.0:
            lo.add([v.index], 1.0 - w, "REPLACE")
        if w > 0.0:
            up.add([v.index], w, "REPLACE")
    ob.modifiers.new("Armature", "ARMATURE").object = arm
    for bone in ("lower", "upper"):
        arm.pose.bones[bone].rotation_mode = "XYZ"
        K.key_vec(arm, f'pose.bones["{bone}"].rotation_euler', frames,
                  [tuple(math.radians(a) for a in e) for e in p[bone]], interp="LINEAR")
    place(light("light", p["light"]), *p["place"])
    return {"cameras": cameras(outs, *p["camera"], lens=p.get("lens", 45.0))}


def build_blur(p, frames, outs):
    floor()
    ball = sphere("mover", 0.4, (0.0, 0.0, 0.4), shader("mover", "diffuse", {"Color": (0.85, 0.3, 0.2)}))
    K.key_vec(ball, "location", [k[0] for k in p["ball"]], [k[1] for k in p["ball"]], interp="LINEAR")
    bar = box("spinner", (1.6, 0.12, 0.12), p["bar_at"], shader("spinner", "diffuse", {"Color": (0.2, 0.45, 0.85)}))
    K.key_vec(bar, "rotation_euler", [k[0] for k in p["bar"]], [(0.0, 0.0, math.radians(k[1])) for k in p["bar"]],
              interp="LINEAR")
    place(light("light", p["light"]), *p["place"])
    cams = cameras(outs, p["camera"][0][1], p["camera"][0][2], lens=p.get("lens", 40.0))
    for cam in cams.values():
        key_places(cam, [k[0] for k in p["camera"]], [(k[1], k[2]) for k in p["camera"]], interp="LINEAR")
    return {"cameras": cams}

def _sectors(n, r):
    """A bmesh fill of a disc of radius r on the XY plane in n sectors, alternating material slots 0 and 1."""
    def fill(bm, uv):
        c = bm.verts.new((0.0, 0.0, 0.0))
        rim = [bm.verts.new((r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0)) for i in range(n)]
        for i in range(n):
            f = bm.faces.new((c, rim[i], rim[(i + 1) % n]))
            f.material_index = i % 2
    return fill


def _arm(spec, frames):
    """An arm: a tube (radius r, length h) along +Z from `at`, skinned to a lower and an upper bone (smoothstep weights
    round the elbow), each bone's swing about the camera's axis keyed LINEAR (`lower`, `upper`: [[frame, degrees]])."""
    arm_data = bpy.data.armatures.new("arm")
    arm = _link(bpy.data.objects.new("arm", arm_data))
    arm.location = spec["at"]
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    h = spec["h"]
    lower = arm_data.edit_bones.new("lower")
    lower.head, lower.tail = (0.0, 0.0, 0.0), (0.0, 0.0, h / 2)
    upper = arm_data.edit_bones.new("upper")
    upper.head, upper.tail = (0.0, 0.0, h / 2), (0.0, 0.0, h)
    upper.parent, upper.use_connect = lower, True
    bpy.ops.object.mode_set(mode="OBJECT")
    ob = tube("arm_skin", spec["r"], h, shader("arm_skin", "diffuse", {"Color": tuple(spec.get("color", (0.8, 0.6, 0.45)))}),
              (0.0, 0.0, 0.0), segments=48, rings=24)
    ob.parent = arm
    lo, up = ob.vertex_groups.new(name="lower"), ob.vertex_groups.new(name="upper")
    blend = spec.get("blend", 0.12)
    for v in ob.data.vertices:
        t = min(1.0, max(0.0, (v.co.z - (h / 2 - blend)) / (2 * blend)))
        w = t * t * (3 - 2 * t)
        if w < 1.0:
            lo.add([v.index], 1.0 - w, "REPLACE")
        if w > 0.0:
            up.add([v.index], w, "REPLACE")
    ob.modifiers.new("Armature", "ARMATURE").object = arm
    for bone in ("lower", "upper"):
        pb = arm.pose.bones[bone]
        pb.rotation_mode = "XYZ"
        # a bone along +Z turns about its own Z (the world's -Y, the camera's axis): a swing across the picture
        K.key_vec(arm, f'pose.bones["{bone}"].rotation_euler', [k[0] for k in spec[bone]],
                  [(0.0, 0.0, math.radians(k[1])) for k in spec[bone]], interp="LINEAR")
    return arm


def build_motion(p, frames, outs):
    """Things moving for the motion blur cases, over a floor and a checkered wall (`wall`: [y, cell], static):
    `objects` (kind card: an emission square facing the camera, sphere, box, tube, sectors: a disc of alternating
    sectors facing the camera; at, size / r / h, color, `keys` {location: [[frame, [x, y, z]]], rotation: [[frame,
    [x, y, z] degrees]]} LINEAR), `arm` (_arm), one light (`light` at `place`) and the camera keyed LINEAR (`camera`:
    [[frame, at, target]])."""
    floor()
    if p.get("wall"):
        y, cell = p["wall"]
        mats = [shader("wall_light", "diffuse", {"Color": (0.7, 0.7, 0.7)}),
                shader("wall_dark", "diffuse", {"Color": (0.12, 0.12, 0.14)})]
        _mesh("wall", checker_fill(int(14 / cell), int(6 / cell), cell), mats, smooth=False, at=(0.0, y, 3.0),
              rot=(math.pi / 2, 0.0, 0.0))
    for o in p.get("objects") or []:
        col = tuple(o.get("color", (0.8, 0.8, 0.8)))
        kind = o["kind"]
        if kind == "card":
            ob = upright(o["name"], o["size"], o["size"], shader(o["name"], "emission", {"Color": col}), tuple(o["at"]))
        elif kind == "sphere":
            ob = sphere(o["name"], o["r"], tuple(o["at"]), shader(o["name"], "diffuse", {"Color": col}))
        elif kind == "box":
            ob = box(o["name"], o["size"], tuple(o["at"]), shader(o["name"], "diffuse", {"Color": col}))
        elif kind == "tube":
            ob = tube(o["name"], o["r"], o["h"], shader(o["name"], "diffuse", {"Color": col}), tuple(o["at"]))
        else:
            mats = [shader(f"{o['name']}_a", "diffuse", {"Color": col}),
                    shader(f"{o['name']}_b", "diffuse", {"Color": tuple(o.get("color_b", (0.9, 0.9, 0.9)))})]
            ob = _mesh(o["name"], _sectors(o.get("n", 24), o["r"]), mats, smooth=False, at=tuple(o["at"]),
                       rot=(math.pi / 2, 0.0, 0.0))
        keys = o.get("keys") or {}
        if "location" in keys:
            K.key_vec(ob, "location", [k[0] for k in keys["location"]], [tuple(k[1]) for k in keys["location"]],
                      interp="LINEAR")
        if "rotation" in keys:
            K.key_vec(ob, "rotation_euler", [k[0] for k in keys["rotation"]],
                      [tuple(math.radians(a) for a in k[1]) for k in keys["rotation"]], interp="LINEAR")
    if p.get("arm"):
        _arm(p["arm"], frames)
    place(light("light", p["light"]), *p["place"])
    cams = cameras(outs, p["camera"][0][1], p["camera"][0][2], lens=p.get("lens", 40.0))
    if len(p["camera"]) > 1:
        for cam in cams.values():
            key_places(cam, [k[0] for k in p["camera"]], [(k[1], k[2]) for k in p["camera"]], interp="LINEAR")
    return {"cameras": cams}



def build_dof(p, frames, outs):
    """Upright emission checkerboards facing the camera (`cards`: name, at, cells [nx, nz], cell, two colours, spin
    degrees in their plane), small emission spheres (`dots`: at, r, colour, strength) and one camera per output with
    depth of field, its f-stop, blades, rotation (degrees), ratio and focus distance keyed per frame (`keys`)."""
    for c in p["cards"]:
        mats = [shader(f"{c['name']}_{i}", "emission", {"Color": col}) for i, col in enumerate(c["colors"])]
        _mesh(c["name"], checker_fill(*c["cells"], c["cell"]), mats, smooth=False, at=c["at"],
              rot=(math.pi / 2, math.radians(c.get("spin", 0.0)), 0.0))
    for i, d in enumerate(p["dots"]):
        sphere(f"dot_{i}", d["r"], d["at"], shader(f"dot_{i}", "emission", {"Color": d["color"],
                                                                            "Strength": d["strength"]}), u=24, v=12)
    cams = cameras(outs, *p["camera"], lens=p.get("lens", 50.0))
    keys = p["keys"]
    for cam in cams.values():
        cam.data.dof.use_dof = True
        for col, (prop, conv) in enumerate((("aperture_fstop", float), ("aperture_blades", int),
                                            ("aperture_rotation", math.radians), ("aperture_ratio", float),
                                            ("focus_distance", float))):
            K.key_prop(cam.data, f"dof.{prop}", frames, [conv(k[col]) for k in keys], interp="CONSTANT")
    return {"cameras": cams}


# ================================================================================================= the world
def _key_socket(nt, sock, frames, values):
    """Key a node socket's value on `frames` (a number, or a colour / vector, per frame), held between keys."""
    path = sock.path_from_id("default_value")
    rows = [list(v) if isinstance(v, (list, tuple)) else [v] for v in values]
    if sock.type == "RGBA":
        rows = [r + [1.0] * (4 - len(r)) for r in rows]
    for c in range(len(rows[0])):
        K.set_fcurve(nt, path, c, frames, [r[c] for r in rows], interp="CONSTANT")


def _socket(node, identifier):
    return next(s for s in node.inputs if s.identifier == identifier)


def _out(node, identifier):
    return next(s for s in node.outputs if s.identifier == identifier)


def _sky(nt, spec, d):
    """The sky's colour from the direction `d` (a socket): `ground` under the horizon, `horizon` rising to `zenith`
    (a square-root ramp of the height), and a `sun` disc ({dir, color, strength, size: degrees}) over it."""
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(d, sep.inputs[0])
    rise = nt.nodes.new("ShaderNodeMath")
    rise.operation, rise.use_clamp = "SQRT", True
    nt.links.new(sep.outputs["Z"], rise.inputs[0])
    up = nt.nodes.new("ShaderNodeMix")
    up.data_type = "RGBA"
    nt.links.new(rise.outputs[0], _socket(up, "Factor_Float"))
    _set(_socket(up, "A_Color"), spec.get("horizon", (0.6, 0.65, 0.8)))
    _set(_socket(up, "B_Color"), spec.get("zenith", (0.15, 0.3, 0.9)))
    edge = nt.nodes.new("ShaderNodeMapRange")
    edge.interpolation_type = "SMOOTHSTEP"
    nt.links.new(sep.outputs["Z"], edge.inputs["Value"])
    edge.inputs["From Min"].default_value, edge.inputs["From Max"].default_value = -0.04, 0.04
    ground = nt.nodes.new("ShaderNodeMix")
    ground.data_type = "RGBA"
    nt.links.new(edge.outputs["Result"], _socket(ground, "Factor_Float"))
    _set(_socket(ground, "A_Color"), spec.get("ground", (0.08, 0.07, 0.06)))
    nt.links.new(_out(up, "Result_Color"), _socket(ground, "B_Color"))
    col = _out(ground, "Result_Color")
    sun = spec.get("sun")
    if sun:
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        nt.links.new(d, dot.inputs[0])
        dot.inputs[1].default_value = Vector(sun["dir"]).normalized()
        disc = nt.nodes.new("ShaderNodeMapRange")
        disc.interpolation_type = "SMOOTHSTEP"
        nt.links.new(dot.outputs["Value"], disc.inputs["Value"])
        half = math.radians(float(sun["size"])) * 0.5
        disc.inputs["From Min"].default_value = math.cos(half * 1.1)
        disc.inputs["From Max"].default_value = math.cos(half * 0.9)
        disc.inputs["To Max"].default_value = float(sun["strength"])
        add = nt.nodes.new("ShaderNodeMix")
        add.data_type, add.blend_type = "RGBA", "ADD"
        nt.links.new(disc.outputs["Result"], _socket(add, "Factor_Float"))
        add.clamp_factor = False
        nt.links.new(col, _socket(add, "A_Color"))
        _set(_socket(add, "B_Color"), sun.get("color", (1.0, 0.95, 0.85)))
        col = _out(add, "Result_Color")
    return col


def _stripes(nt, spec, d):
    """`bands` vertical bands round the up axis, `a` and `b` in turn (sharp edges for the glossy lobes to blur), over
    the sky's ground under the horizon."""
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(d, sep.inputs[0])
    ang = nt.nodes.new("ShaderNodeMath")
    ang.operation = "ARCTAN2"
    nt.links.new(sep.outputs["Y"], ang.inputs[0])
    nt.links.new(sep.outputs["X"], ang.inputs[1])
    turns = nt.nodes.new("ShaderNodeMath")
    turns.operation = "MULTIPLY"
    nt.links.new(ang.outputs[0], turns.inputs[0])
    turns.inputs[1].default_value = float(spec.get("bands", 12)) / (2.0 * math.pi)
    frac = nt.nodes.new("ShaderNodeMath")
    frac.operation = "FRACT"
    nt.links.new(turns.outputs[0], frac.inputs[0])
    step = nt.nodes.new("ShaderNodeMath")
    step.operation = "GREATER_THAN"
    nt.links.new(frac.outputs[0], step.inputs[0])
    step.inputs[1].default_value = 0.5
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(step.outputs[0], _socket(mix, "Factor_Float"))
    _set(_socket(mix, "A_Color"), spec["a"])
    _set(_socket(mix, "B_Color"), spec["b"])
    edge = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(sep.outputs["Z"], edge.inputs["Value"])
    edge.inputs["From Min"].default_value, edge.inputs["From Max"].default_value = -0.02, 0.02
    ground = nt.nodes.new("ShaderNodeMix")
    ground.data_type = "RGBA"
    nt.links.new(edge.outputs["Result"], _socket(ground, "Factor_Float"))
    _set(_socket(ground, "A_Color"), spec.get("ground", (0.08, 0.07, 0.06)))
    nt.links.new(_out(mix, "Result_Color"), _socket(ground, "B_Color"))
    return _out(ground, "Result_Color")


def world(spec, frames):
    """The case's world from its spec (`kind`): uniform (`color`), sky (_sky), stripes (_stripes) or lightpath (`camera`
    to camera rays, `light` to every other ray: what lights the scene); a Background of that colour times `strength`,
    both keyable per frame (`keys`: {"color": [...], "strength": [...]}); `props` sets World properties (sun_threshold,
    probe_resolution, use_sun_shadow, ...) and `visibility` its ray visibility flags (cycles_visibility)."""
    w = bpy.data.worlds.new("lab_world")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = float(spec.get("strength", 1.0))
    kind = spec["kind"]
    if kind == "uniform":
        _set(bg.inputs["Color"], spec["color"])
    else:
        d = nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
        if kind == "sky":
            nt.links.new(_sky(nt, spec, d), bg.inputs["Color"])
        elif kind == "stripes":
            nt.links.new(_stripes(nt, spec, d), bg.inputs["Color"])
        elif kind == "lightpath":
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            nt.links.new(nt.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"], _socket(mix, "Factor_Float"))
            _set(_socket(mix, "A_Color"), spec["light"])
            _set(_socket(mix, "B_Color"), spec["camera"])
            nt.links.new(_out(mix, "Result_Color"), bg.inputs["Color"])
        else:
            raise ValueError(f"world kind {kind}")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    for key, values in (spec.get("keys") or {}).items():
        _key_socket(nt, bg.inputs[key.capitalize()], frames, values)
    for k, v in (spec.get("props") or {}).items():
        setattr(w, k, v)
    for k, v in (spec.get("visibility") or {}).items():
        setattr(w.cycles_visibility, k, v)
    return w


def scene_material(name, spec):
    """A one-BSDF material (`kind`, `inputs`, `distribution`: shader()) whose inputs `keys` ({input: [value per
    frame]}) change per frame, held between keys."""
    m = shader(spec.get("name", name), spec.get("kind", "diffuse"), spec.get("inputs"), spec.get("distribution"))
    node = next(n for n in m.node_tree.nodes if n.type != "OUTPUT_MATERIAL")
    for key, values in (spec.get("keys") or {}).items():
        _key_socket(m.node_tree, node.inputs[key], list(range(1, len(values) + 1)), values)
    return m


def build_scene(p, frames, outs):
    """Objects lit by the case's world and its lights: `objects` (kind box, sphere, plate or tube, as build_shadow's:
    name, at, size / r / h, yaw, rot, camera: false) each with its `material` (scene_material; grey diffuse by
    default), `lights` (light specs, each aimed from `place` [at, target]), and one camera per output keyed per frame
    to `camera` ([[at, target], ...], one per frame, or one for all) through a `lens`."""
    for o in p["objects"]:
        mat = scene_material(o["name"], o.get("material") or {"kind": "diffuse", "inputs": {"Color": (0.7, 0.7, 0.7)}})
        if o["kind"] == "box":
            ob = box(o["name"], o["size"], o["at"], mat, o.get("yaw", 0.0))
        elif o["kind"] == "sphere":
            ob = sphere(o["name"], o["r"], o["at"], mat)
        elif o["kind"] == "plate":
            ob = plane(o["name"], o["size"][0], o["size"][1], mat, at=o["at"])
        else:
            ob = tube(o["name"], o["r"], o["h"], mat, o["at"])
        if "rot" in o:
            ob.rotation_euler = [math.radians(a) for a in o["rot"]]
        ob.visible_camera = bool(o.get("camera", True))
    for i, spec in enumerate(p.get("lights") or []):
        place(light(spec.get("name", f"light_{i}"), spec), *spec["place"])
    views = p["camera"] if isinstance(p["camera"][0][0], (list, tuple)) else [p["camera"]]
    cams = cameras(outs, *views[0], lens=p.get("lens", 35.0))
    if len(views) > 1:
        for cam in cams.values():
            key_places(cam, frames[:len(views)], [tuple(v) for v in views])
    return {"cameras": cams}


# ================================================================================================= MMD kinds
def _strip_anim(idb):
    """Leave `idb` unanimated but driven: its action and NLA strips go, its drivers stay."""
    ad = getattr(idb, "animation_data", None)
    if ad is None:
        return
    ad.action = None
    for t in list(ad.nla_tracks):
        ad.nla_tracks.remove(t)


def _member_at_rest(armature):
    """Strip the open scene to one cast member (its MMD root, armature, meshes and morph holder), unanimated, at its
    rest pose, at the origin facing -Y: every other object, marker and mk scene property goes. Returns (armature,
    meshes, the names kept)."""
    sc = bpy.context.scene
    arm = S.find_armature(armature)
    root = arm
    while root.parent is not None:
        root = root.parent
    meshes = S.model_meshes(arm)
    keep = {root.name, arm.name, *(m.name for m in meshes),
            *(c.name for c in root.children_recursive if c.type == "MESH" and getattr(c, "mmd_type", "") == "PLACEHOLDER")}
    for o in list(bpy.data.objects):
        if o.name not in keep:
            bpy.data.objects.remove(o, do_unlink=True)
    for name in keep:
        o = bpy.data.objects[name]
        _strip_anim(o)
        if o.type == "MESH" and o.data.shape_keys is not None:
            _strip_anim(o.data.shape_keys)
            for kb in o.data.shape_keys.key_blocks[1:]:
                kb.value = 0.0
    for m in {s.material for o in meshes for s in o.material_slots if s.material is not None}:
        _strip_anim(m)
        _strip_anim(m.node_tree)
    root.location = (0.0, 0.0, 0.0)
    root.rotation_mode = "XYZ"
    root.rotation_euler = (0.0, 0.0, 0.0)
    arm.data.pose_position = "REST"
    for mk in list(sc.timeline_markers):
        sc.timeline_markers.remove(mk)
    for k in [k for k in sc.keys() if k.startswith("mk_")]:
        del sc[k]
    sc.use_nodes = False
    bpy.context.view_layer.update()
    return arm, meshes, keep


def _member_lights(p, frames, focus):
    """One light per frame (p["lights"], each at its offset from `focus`, aimed at it), then all of them at once."""
    obs = []
    for spec in p["lights"]:
        ob = light(spec["name"], spec)
        place(ob, tuple(Vector(focus) + Vector(spec["offset"])), focus)
        obs.append(ob)
    show_on([[ob] for ob in obs] + [obs] * (len(frames) - len(obs)), frames)


def build_mmd_model(p, frames, outs, armature):
    arm, meshes, _ = _member_at_rest(armature)
    head = arm.matrix_world @ arm.data.bones[S.resolve_bone(arm, "head")].head_local
    dg = bpy.context.evaluated_depsgraph_get()
    zs = []
    for o in meshes:
        ev = o.evaluated_get(dg)
        zs += [(ev.matrix_world @ Vector(c)).z for c in ev.bound_box]
    lo, hi = min(zs), max(zs)
    _member_lights(p, frames, (0.0, 0.0, lo + 0.68 * (hi - lo)))
    lens, cams = p.get("lens", 50.0), {}
    for o in outs:
        if p["views"][o] == "full":                                   # head to toe, a fifth to spare
            mid, dist = (lo + hi) / 2, 0.62 * (hi - lo) * lens / 18.0
            cams[o] = camera(f"{SHOT}@{o}", (0.0, -dist, mid), (0.0, 0.0, mid), lens)
        else:                                                         # the face: a little over the head bone's root
            eye = head + Vector((0.0, 0.0, p.get("eye", 0.07)))
            at = eye + Vector((0.0, -p.get("face_distance", 0.9), 0.02))
            cams[o] = camera(f"{SHOT}@{o}", tuple(at), tuple(eye), lens)
    return {"cameras": cams, "member": {"armature": arm.name, "meshes": [m.name for m in meshes],
                                        "height": round(hi - lo, 4), "head": [round(x, 4) for x in head]}}


def build_mmd_sphere(p, frames, outs, armature):
    _, meshes, keep = _member_at_rest(armature)
    faces = {}
    for o in meshes:
        slots = [s.material for s in o.material_slots]
        for poly in o.data.polygons:
            m = slots[poly.material_index] if poly.material_index < len(slots) else None
            if m is not None:
                faces[m.name] = faces.get(m.name, 0) + 1
    names = p.get("materials") or sorted(faces, key=lambda n: -faces[n])[:len(frames)]
    mats = []
    for n in names:
        m = bpy.data.materials[n].copy()
        m.animation_data_clear()                       # its morph drivers read the member's sliders, which go
        if m.node_tree is not None:
            m.node_tree.animation_data_clear()
        mats.append(m)
    for name in keep:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    centre = Vector((0.0, 0.0, 1.0))
    balls = [[sphere(f"ball_{i:02d}", p.get("r", 0.5), tuple(centre), m)] for i, m in enumerate(mats)]
    show_on(balls + [balls[-1]] * (len(frames) - len(balls)), frames)
    for spec in p["lights"]:
        place(light(spec["name"], spec), tuple(centre + Vector(spec["offset"])), tuple(centre))
    return {"cameras": cameras(outs, *p["camera"], lens=p.get("lens", 50.0)),
            "materials": {n: faces.get(n, 0) for n in names}}


KINDS = {"light": build_light, "shadow": build_shadow, "spheres": build_spheres, "ramp": build_ramp, "edge": build_edge,
         "alpha": build_alpha, "bend": build_bend, "blur": build_blur, "motion": build_motion, "dof": build_dof,
         "scene": build_scene, "mmd_model": build_mmd_model, "mmd_sphere": build_mmd_sphere}


# ================================================================================================= scene, frames, bake
def _clear():
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for mk in list(sc.timeline_markers):
        sc.timeline_markers.remove(mk)


def _eevee(e, table):
    """Set scene.eevee properties from a table, a nested table setting a struct's own (ray_tracing_options)."""
    for k, v in table.items():
        if not hasattr(e, k) and not (isinstance(v, dict)):
            continue              # a property 4.2 had and 5.x dropped (fast_gi_thickness_far): not settable here
        if isinstance(v, dict):
            _eevee(getattr(e, k), v)
        else:
            setattr(e, k, v)


def _settings(sc, case):
    """The render settings of a lab case: EEVEE Next at the case's samples, no world (black, no ambient) unless the
    case has one (`world`: world()), the film (opaque unless `render` says film_transparent) and the view transform as
    mk's scenes have them unless the case says otherwise, motion blur only when it asks; the case's `eevee` table sets
    any scene.eevee property over those (shadow_step_count, use_raytracing, ray_tracing_options: {...}, ...)."""
    r, e, rs = sc.render, sc.eevee, case.get("render") or {}
    K.set_engine(sc, "eevee")
    sc.world = world(case["world"], list(range(1, int(case["frames"]) + 1))) if case.get("world") else None
    r.fps, r.fps_base = int(case.get("fps", 30)), 1.0
    sc.frame_start, sc.frame_end = 1, int(case["frames"])
    r.resolution_percentage = 100
    r.filter_size = float(rs.get("filter_size", 1.5))
    r.film_transparent = bool(rs.get("film_transparent", False))
    r.dither_intensity = float(rs.get("dither", 1.0))
    e.taa_render_samples = int(case.get("samples", 64))
    e.use_shadows, e.shadow_ray_count, e.shadow_step_count = True, 1, 6
    e.use_raytracing = False
    e.light_threshold = 0.01
    _eevee(e, case.get("eevee") or {})
    vs = sc.view_settings
    vs.view_transform = rs.get("view", "AgX")
    vs.look = rs.get("look", "None")
    vs.exposure, vs.gamma, vs.use_curve_mapping = float(rs.get("exposure", 0.0)), 1.0, False
    sc.display_settings.display_device = "sRGB"
    mb = case.get("motion_blur")
    r.use_motion_blur = mb is not None
    if mb is not None:
        mb = mb if isinstance(mb, dict) else {"shutter": mb}
        r.motion_blur_shutter = float(mb["shutter"])
        r.motion_blur_position = mb.get("position", "CENTER")
        e.motion_blur_steps = int(mb.get("steps", 1))


def _shot_table(sc, case, cams):
    """The shots stage's table and marker for one shot over the whole case, a camera per output."""
    n = int(case["frames"])
    sc["mk_shots"] = json.dumps([{"name": SHOT, "from": 1, "to": n + 1, "cameras": {o: c.name for o, c in cams.items()},
                                  "keyed": [1, n]}])
    mk = sc.timeline_markers.new(SHOT, frame=1)
    mk.camera = next(iter(cams.values()))
    sc.camera = mk.camera


def _render(sc, case, folders):
    """Every reference frame of every output, rendered once and saved in each form `folders` names ({form: a folder
    holding <output>/}): png8 8-bit RGB as `mk render` writes a frame (view transform, dither), png16 the same picture
    as 16-bit PNG (before the 8-bit quantisation and its dither), exr scene-linear OpenEXR half (no view transform)."""
    r, s = sc.render, sc.render.image_settings
    out, t0 = {}, time.time()
    for o in case["outputs"]:
        S.bind_aspect(o["name"], sc)
        r.resolution_x, r.resolution_y = (int(x) for x in o["size"])
        dirs = {k: os.path.join(v, o["name"]) for k, v in folders.items()}
        for p in dirs.values():
            os.makedirs(p, exist_ok=True)
        for f in case["refs"]:
            sc.frame_set(int(f))
            name = f"{int(f):05d}"
            s.file_format, s.color_mode, s.color_depth, s.compression = "PNG", "RGB", "8", 15
            r.filepath = os.path.join(dirs["png8"], name + ".png")
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images["Render Result"]
            if "png16" in dirs:
                s.color_depth = "16"
                img.save_render(os.path.join(dirs["png16"], name + ".png"), scene=sc)
            if "exr" in dirs:
                s.file_format, s.color_depth, s.exr_codec = "OPEN_EXR", "16", "ZIP"
                img.save_render(os.path.join(dirs["exr"], name + ".exr"), scene=sc)
            s.file_format, s.color_depth = "PNG", "8"
            print(f"LAB {case['name']} {o['name']} frame {f}: {time.time() - t0:.1f} s", flush=True)
        out[o["name"]] = len(case["refs"])
    return {"frames": out, "seconds": round(time.time() - t0, 1)}


def _props(rna):
    """The plain (bool, int, float, single enum) properties of a settings struct, by identifier."""
    return {p.identifier: getattr(rna, p.identifier) for p in rna.bl_rna.properties
            if p.type in ("BOOLEAN", "INT", "FLOAT", "ENUM") and not getattr(p, "is_array", False)
            and not getattr(p, "is_enum_flag", False)}


def _bake():
    """The build's bake stage on the saved scene (beside it: scene.bake/); a failure is the report, not an error."""
    t0 = time.time()
    try:
        from .build import Ctx
        from .build import bake as BK
        rep = BK.run(Ctx({"project": CTX["project"], "config": CTX["config"], "args": {}})) or {}
        with open(os.path.join(os.path.dirname(bpy.data.filepath), "scene.bake", "bake.json"), encoding="utf-8") as fh:
            return dict(rep, version=json.load(fh).get("version"))
    except Exception as e:  # noqa: BLE001 - the frames stay usable; `mk parity lab --rebake` tries again
        import traceback
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:],
                "seconds": round(time.time() - t0, 1)}


@op("lab_build")
def lab_build(args):
    case, d = args["case"], args["dir"]
    sc = bpy.context.scene
    t0 = time.time()
    frames = list(range(1, int(case["frames"]) + 1))
    outs = [o["name"] for o in case["outputs"]]
    if case["kind"].startswith("mmd_"):
        built = KINDS[case["kind"]](case["params"], frames, outs, args["armature"])
    else:
        _clear()
        built = KINDS[case["kind"]](case["params"], frames, outs)
    _settings(sc, case)
    _shot_table(sc, case, built.pop("cameras"))
    sc.frame_set(1)
    blend = os.path.join(d, "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
    t_build = time.time() - t0
    folders = {"png8": os.path.join(d, "ref"), "png16": os.path.join(d, "ref16"), "exr": os.path.join(d, "linear")}
    refs = _render(sc, case, folders) if args.get("render", True) else None
    bake = _bake()
    return {"blend": blend, "build_seconds": round(t_build, 1), "refs": refs, "bake": bake, "built": built,
            "objects": [o.name for o in sc.objects], "eevee": _props(sc.eevee), "blender": bpy.app.version_string,
            "seconds": round(time.time() - t0, 1)}


# ================================================================================================= variants
def _gpu():
    """Point Cycles at a GPU (its compute backend and devices, for this session only: preferences are not saved):
    {"backend", "devices"}, or None when Blender finds none."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return None
    for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            prefs.compute_device_type = kind
        except TypeError:                                  # not a backend of this build
            continue
        prefs.refresh_devices()
        found = [dev for dev in prefs.devices if dev.type == kind]
        if found:
            for dev in prefs.devices:
                dev.use = dev.type == kind
            return {"backend": kind, "devices": [dev.name for dev in found]}
    prefs.compute_device_type = "NONE"
    return None


def _cycles(sc, v, lights):
    """Cycles as the truth EEVEE's direct light approximates: no bounce (EEVEE without a world, light probes or ray
    tracing has no indirect light; v["bounces"] diffuse and glossy bounces for the indirect light a world, ray tracing
    and fast GI approximate) but through transparency (v["transparent"] layers, no Russian roulette on them),
    v["samples"] fixed samples from seed v["seed"], no adaptive sampling, denoiser, clamping or light threshold, a
    Blackman-Harris film filter as wide as EEVEE's, the lights unseen by the camera (EEVEE never draws them); the view
    transform and dither stay the scene's. On a GPU unless v["device"] is "CPU" or Blender finds none."""
    r, cy = sc.render, sc.cycles
    r.engine = "CYCLES"
    gpu = _gpu() if v.get("device", "GPU") == "GPU" else None
    cy.device = "GPU" if gpu else "CPU"
    cy.samples, cy.seed, cy.use_animated_seed = int(v["samples"]), int(v.get("seed", 0)), False
    cy.use_adaptive_sampling, cy.use_denoising = False, False
    for vl in sc.view_layers:
        vl.cycles.use_denoising = False
    bounces = int(v.get("bounces", 0))
    cy.max_bounces = cy.diffuse_bounces = cy.glossy_bounces = bounces
    cy.transmission_bounces = cy.volume_bounces = 0
    cy.transparent_max_bounces = cy.min_transparent_bounces = int(v.get("transparent", 64))
    cy.sample_clamp_direct = cy.sample_clamp_indirect = 0.0
    cy.light_sampling_threshold, cy.blur_glossy = 0.0, 0.0
    cy.use_guiding, cy.auto_scrambling_distance, cy.scrambling_distance = False, False, 1.0
    cy.filter_width, cy.pixel_filter_type, cy.film_exposure = r.filter_size, "BLACKMAN_HARRIS", 1.0
    for ob in lights:
        ob.visible_camera = False
    return {"device": gpu or "CPU", "cycles": _props(cy)}


@op("lab_variant")
def lab_variant(args):
    """A variant of a built case (args: "case" its spec, "variant" {name, engine EEVEE|CYCLES, shadows, samples, seed,
    transparent, device, eevee: scene.eevee properties set over the case's}, "out" its folder): the job opens the case's
    saved scene, changes it in memory and renders every reference frame of every output into out/<output>/ (8-bit PNG
    and scene-linear OpenEXR half beside it, as lab_build's ref/ and linear/) and out/16/<output>/ (16-bit PNG).
    Nothing is saved: scene.blend, ref/ and scene.bake/ stay as they are."""
    case, v, out = args["case"], args["variant"], args["out"]
    sc = bpy.context.scene
    t0 = time.time()
    lights = [ob for ob in sc.objects if ob.type == "LIGHT"]
    if not v.get("shadows", True):
        for ob in lights:
            ob.data.use_shadow = False
    if v["engine"] == "CYCLES":
        settings = _cycles(sc, v, lights)
    else:
        if v.get("samples"):
            sc.eevee.taa_render_samples = int(v["samples"])
        _eevee(sc.eevee, v.get("eevee") or {})
        settings = {"eevee": _props(sc.eevee)}
    frames = _render(sc, case, {"png8": out, "png16": os.path.join(out, "16"), "exr": out})
    n = sum(frames["frames"].values())
    return {"engine": sc.render.engine, **settings, "frames": frames["frames"], "render_seconds": frames["seconds"],
            "s_per_frame": round(frames["seconds"] / max(1, n), 2),
            "lights": {ob.name: {"type": ob.data.type, "use_shadow": ob.data.use_shadow} for ob in lights},
            "blender": bpy.app.version_string, "seconds": round(time.time() - t0, 1)}
