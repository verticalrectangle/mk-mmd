"""Shared kit of the cafe props and the cafe set: palette-slot colours, bmesh sweeps and lathes, shader-node shorthands
and driven parameters. A helper module: it registers no builder.

Colours. Every colour is a blend of palette slots, mixed in linear light: `K.blend(gold=.6, base=.4)` (weights are
normalised; `k` scales the brightness, `hue` (deg) and `chroma` tilt it in OKLab, `a` is the alpha). `K.slot("pine")` is
one slot, `K.light(...)` a light tint (a blend normalised to its brightest channel). The slots come from the project's palette; `[[prop]] slots = {pine = "#123456"}` overrides single slots for one
prop. Without a palette (the build stage passes only the overrides) the Rose Pine Dawn palette is used, which is what the
blends were fitted on, so `[look] palette = "rose-pine-dawn"` reproduces the reference look and other palettes recolour
the props consistently.

Parameters. A prop that has something to animate (steam, bulb gain) exposes it as a custom property on its root empty;
`K.param(nt, "steam")` returns the output of a Value node in a material whose value a driver copies from that property,
so a project keys the property and the shader follows. `K.clock(nt)` is the scene time in seconds (frame / fps).

Frames. A prop is built in its own frame (the root empty): its documented origin is the contact point on the floor or on
the table top, +Z up. Everything is named `<prop name>_<part>` so several instances coexist."""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

from ..mesh import as_collider
from .cafe_colors import DAWN, Colors, mix, shade  # noqa: F401  (re-exported for the builders)


# ================================================================= curves -> points
def catmull(pts, n=8, closed=False):
    """Uniform Catmull-Rom through `pts` (iterables of 3 floats) -> list[Vector], n samples per span."""
    P = [Vector(p) for p in pts]
    m = len(P)
    out = []
    spans = m if closed else m - 1
    for i in range(spans):
        p0 = P[(i - 1) % m] if (closed or i > 0) else P[0] * 2 - P[1]
        p1 = P[i]
        p2 = P[(i + 1) % m]
        p3 = P[(i + 2) % m] if (closed or i + 2 < m) else P[-1] * 2 - P[-2]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-1].copy())
    return out


def path_length(pts):
    return sum((pts[i + 1] - pts[i]).length for i in range(len(pts) - 1))


def circle_pts(r, n=48, z=0.0, c=(0.0, 0.0), a0=0.0):
    """Horizontal circle (closed path, n points, no repeated end point)."""
    return [Vector((c[0] + r * math.cos(a0 + 2 * math.pi * i / n), c[1] + r * math.sin(a0 + 2 * math.pi * i / n), z))
            for i in range(n)]


def _perp(t):
    a = Vector((1, 0, 0)) if abs(t.x) < 0.9 else Vector((0, 1, 0))
    return (a - t * a.dot(t)).normalized()


# ================================================================= tube sweep
def bm_tube(bm, pts, radius, sides=12, mat=0, cap="flat", closed=False, profile=(1.0, 1.0), up=(0, 0, 1),
            smooth=True, uv=True, phase=0.0):
    """Sweep a (elliptical) circle along the polyline `pts` (list of Vector; give a dense, smooth polyline).
    radius: float | list per point | callable(t in 0..1).  profile=(a, b): ellipse factors along the frame normal N (the
    projection of `up` onto the plane perpendicular to the path) and binormal.  cap: 'flat' | 'round' | 'none'
    (cap=(start, end) tuple of those also accepted).  closed=True makes a ring (no caps).
    UVs: u = arc length along the path (metres), v = around the section (metres of circumference), seam at phase.
    Rotation-minimising frames; returns the list of rings (lists of BMVert)."""
    pts = [Vector(p) for p in pts]
    n = len(pts)
    if callable(radius):
        rad = [radius(i / max(n - 1, 1)) for i in range(n)]
    elif isinstance(radius, (list, tuple)):
        rad = list(radius)
    else:
        rad = [float(radius)] * n
    cap_s, cap_e = (cap, cap) if isinstance(cap, str) else cap
    if closed:
        cap_s = cap_e = "none"
    # round caps: extend the polyline with shrinking rings
    if cap_s == "round" and n > 1:
        t0 = (pts[0] - pts[1]).normalized()
        r0 = rad[0]
        ext = [(pts[0] + t0 * r0 * math.sin(f), r0 * math.cos(f)) for f in (math.radians(80), math.radians(60), math.radians(30))]
        pts = [e[0] for e in ext] + pts
        rad = [e[1] for e in ext] + rad
    if cap_e == "round" and n > 1:
        t1 = (pts[-1] - pts[-2]).normalized()
        r1 = rad[-1]
        base = pts[-1]
        for f in (math.radians(30), math.radians(60), math.radians(80)):
            pts.append(base + t1 * r1 * math.sin(f))
            rad.append(r1 * math.cos(f))
    n = len(pts)
    # tangents
    T = []
    for i in range(n):
        if closed:
            a = pts[(i + 1) % n] - pts[(i - 1) % n]
        else:
            a = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        T.append(a.normalized())
    # rotation-minimising frames
    upv = Vector(up)
    N0 = upv - T[0] * upv.dot(T[0])
    N0 = N0.normalized() if N0.length > 1e-6 else _perp(T[0])
    Ns = [N0]
    for i in range(1, n):
        p = Ns[-1] - T[i] * Ns[-1].dot(T[i])
        Ns.append(p.normalized() if p.length > 1e-8 else _perp(T[i]))
    if closed:       # distribute the leftover twist so the ring closes without a seam
        pe = Ns[-1] - T[0] * Ns[-1].dot(T[0])
        pe = pe.normalized()
        ang = math.atan2(pe.cross(Ns[0]).dot(T[0]), pe.dot(Ns[0]))
        for i in range(n):
            Ns[i] = Matrix.Rotation(ang * i / n, 3, T[i]) @ Ns[i]
    a_, b_ = profile
    rings = []
    for i in range(n):
        N = Ns[i]
        B = T[i].cross(N)
        ring = []
        for j in range(sides):
            a = phase + 2 * math.pi * j / sides
            ring.append(bm.verts.new(pts[i] + N * (math.cos(a) * rad[i] * a_) + B * (math.sin(a) * rad[i] * b_)))
        rings.append(ring)
    # arc length
    s = [0.0]
    for i in range(1, n):
        s.append(s[-1] + (pts[i] - pts[i - 1]).length)
    uvl = bm.loops.layers.uv.verify() if uv else None
    spans = n if closed else n - 1
    for i in range(spans):
        i2 = (i + 1) % n
        circ0 = 2 * math.pi * rad[i] * 0.5 * (a_ + b_)
        circ1 = 2 * math.pi * rad[i2] * 0.5 * (a_ + b_)
        for j in range(sides):
            j2 = (j + 1) % sides
            try:
                f = bm.faces.new((rings[i][j], rings[i][j2], rings[i2][j2], rings[i2][j]))
            except ValueError:
                continue
            f.smooth = smooth
            f.material_index = mat
            if uv:
                s0 = s[i]
                s1 = s[i + 1] if i + 1 < n else s[-1] + (pts[0] - pts[-1]).length
                us = (s0, s0, s1, s1)
                vs = (j / sides * circ0, (j + 1) / sides * circ0, (j + 1) / sides * circ1, j / sides * circ1)
                for lp, u, v in zip(f.loops, us, vs):
                    lp[uvl].uv = (u, v)
    # flat caps (duplicated verts so the rim stays crisp)
    for which, ring, Tc, kind in (("s", rings[0], -T[0], cap_s), ("e", rings[-1], T[-1], cap_e)):
        if kind == "none":
            continue
        dup = [bm.verts.new(v.co) for v in ring]
        order = dup if which == "e" else dup[::-1]
        try:
            f = bm.faces.new(order)
        except ValueError:
            continue
        f.smooth = False
        f.material_index = mat
        if uv:
            c = sum((v.co for v in dup), Vector()) / len(dup)
            rr = max((v.co - c).length for v in dup) or 1e-6
            for lp in f.loops:
                d = lp.vert.co - c
                lp[uvl].uv = (0.5 + 0.5 * d.x / rr, 0.5 + 0.5 * d.y / rr)
    return rings


# ================================================================= lathe
def bm_lathe(bm, prof, segs=48, mat=0, smooth=True, phase=0.0, center=(0.0, 0.0, 0.0), uv=True, mat_ranges=None):
    """Revolve a profile [(r, z), ...] around the local Z axis (normals follow the profile direction: bottom->top on
    the outside gives outward normals; continue over the lip and down the inside for a hollow vessel).  A point with r=0
    becomes a single apex vertex.  Two consecutive identical points make a hard crease (no faces between them).
    UV: u = angle fraction, v = profile arc length (m).  mat_ranges: optional list of (first_profile_idx, mat_index)."""
    cx, cy, cz = center
    rings = []
    for (r, z) in prof:
        if abs(r) < 1e-9:
            rings.append([bm.verts.new((cx, cy, cz + z))])
        else:
            rings.append([bm.verts.new((cx + r * math.cos(phase + 2 * math.pi * j / segs),
                                        cy + r * math.sin(phase + 2 * math.pi * j / segs), cz + z))
                          for j in range(segs)])
    v_acc = [0.0]
    for i in range(1, len(prof)):
        v_acc.append(v_acc[-1] + math.hypot(prof[i][0] - prof[i - 1][0], prof[i][1] - prof[i - 1][1]))
    uvl = bm.loops.layers.uv.verify() if uv else None

    def mat_at(i):
        m = mat
        if mat_ranges:
            for first, mi in mat_ranges:
                if i >= first:
                    m = mi
        return m

    for i in range(len(prof) - 1):
        if abs(prof[i][0] - prof[i + 1][0]) < 1e-9 and abs(prof[i][1] - prof[i + 1][1]) < 1e-9:
            continue
        A, B = rings[i], rings[i + 1]
        for j in range(segs):
            j2 = (j + 1) % segs
            if len(A) == 1 and len(B) == 1:
                continue
            if len(A) == 1:
                vv = (A[0], B[j2], B[j])
                us = ((j + 0.5) / segs, (j + 1) / segs, j / segs)
                vs = (v_acc[i], v_acc[i + 1], v_acc[i + 1])
            elif len(B) == 1:
                vv = (A[j], A[j2], B[0])
                us = (j / segs, (j + 1) / segs, (j + 0.5) / segs)
                vs = (v_acc[i], v_acc[i], v_acc[i + 1])
            else:
                vv = (A[j], A[j2], B[j2], B[j])
                us = (j / segs, (j + 1) / segs, (j + 1) / segs, j / segs)
                vs = (v_acc[i], v_acc[i], v_acc[i + 1], v_acc[i + 1])
            try:
                f = bm.faces.new(vv)
            except ValueError:
                continue
            f.smooth = smooth
            f.material_index = mat_at(i)
            if uv:
                for lp, u, v in zip(f.loops, us, vs):
                    lp[uvl].uv = (u, v)
    return rings


# ================================================================= other bmesh helpers
def bm_quad(bm, p0, p1, p2, p3, mat=0, smooth=False, uvs=None):
    """One quad (counter-clockwise seen from the normal side) with optional UVs [(u,v)]*4."""
    f = bm.faces.new([bm.verts.new(p) for p in (p0, p1, p2, p3)])
    f.material_index = mat
    f.smooth = smooth
    if uvs:
        uvl = bm.loops.layers.uv.verify()
        for lp, uvv in zip(f.loops, uvs):
            lp[uvl].uv = uvv
    return f


def bm_transform(bm, verts, mat4):
    bmesh.ops.transform(bm, matrix=mat4, verts=list(verts))


def bm_append(dst, src, mat4=None, mat_offset=0):
    """Copy all of bmesh `src` into `dst` (optionally transformed); returns the new verts."""
    vmap = {}
    for v in src.verts:
        co = mat4 @ v.co if mat4 is not None else v.co.copy()
        vmap[v] = dst.verts.new(co)
    uv_s = src.loops.layers.uv.active
    uv_d = dst.loops.layers.uv.verify() if uv_s is not None else None
    for f in src.faces:
        try:
            nf = dst.faces.new([vmap[v] for v in f.verts])
        except ValueError:
            continue
        nf.smooth = f.smooth
        nf.material_index = f.material_index + mat_offset
        if uv_s is not None:
            for a, b in zip(f.loops, nf.loops):
                b[uv_d].uv = a[uv_s].uv
    return list(vmap.values())


def bm_uv_sphere(bm, r, loc=(0, 0, 0), seg=16, rings=8, mat=0, scale=(1, 1, 1)):
    g = bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=r)
    vs = g["verts"]
    bmesh.ops.scale(bm, vec=Vector(scale), verts=vs)
    bmesh.ops.translate(bm, vec=Vector(loc), verts=vs)
    fs = {f for v in vs for f in v.link_faces}
    for f in fs:
        f.smooth = True
        f.material_index = mat
    return vs


def bm_box(bm, size, loc=(0, 0, 0), mat=0, rot_z=0.0, smooth=False):
    g = bmesh.ops.create_cube(bm, size=1.0)
    vs = g["verts"]
    bmesh.ops.scale(bm, vec=Vector(size), verts=vs)
    if rot_z:
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(rot_z, 3, "Z"), verts=vs)
    bmesh.ops.translate(bm, vec=Vector(loc), verts=vs)
    for f in {f for v in vs for f in v.link_faces}:
        f.material_index = mat
        f.smooth = smooth
    return vs


def bevel_modifier(o, width=0.002, segments=2, angle=30.0, limit="ANGLE"):
    md = o.modifiers.new("Bevel", "BEVEL")
    md.width = width
    md.segments = segments
    md.limit_method = limit
    md.angle_limit = math.radians(angle)
    md.harden_normals = False
    return md


# ================================================================= shader node shorthands
def N(nt, kind, loc=(0, 0), inputs=None, **attrs):
    """nt.nodes.new(kind) with location, attribute settings and input default values in one call."""
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in attrs.items():
        setattr(n, k, v)
    for k, v in (inputs or {}).items():
        n.inputs[k].default_value = v
    return n


def L(nt, a, b):
    nt.links.new(a, b)


def principled(nt, out, loc=(600, 0), **inp):
    """Add a Principled BSDF wired to `out`'s Surface; inp = {'Base Color': rgba, 'Roughness': .., 'Coat Weight': ..}
    (underscores stand for spaces: Coat_Weight)."""
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.location = loc
    for k, v in inp.items():
        b.inputs[k.replace("_", " ")].default_value = v
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return b


def ramp(nt, loc, stops, interp="LINEAR"):
    """ColorRamp with stops [(pos, rgba), ...] (positions 0..1)."""
    r = nt.nodes.new("ShaderNodeValToRGB")
    r.location = loc
    r.color_ramp.interpolation = interp
    cr = r.color_ramp
    while len(cr.elements) > len(stops):
        cr.elements.remove(cr.elements[-1])
    while len(cr.elements) < len(stops):
        cr.elements.new(0.5)
    for el, (p, c) in zip(cr.elements, stops):
        el.position, el.color = p, c
    return r


def clear_drivers(idb):
    ad = getattr(idb, "animation_data", None)
    if ad is not None:
        for fc in list(ad.drivers):
            ad.drivers.remove(fc)


def drive(id_data, data_path, expr, index=-1, var=None):
    """Scripted driver `expr` on a property. `var` = (name, id, data_path) adds a single-property variable that reads
    a custom property (id_type OBJECT). Without var the expression may use the builtin `frame`."""
    res = id_data.driver_add(data_path, index) if index >= 0 else id_data.driver_add(data_path)
    fcs = res if isinstance(res, list) else [res]
    for fc in fcs:
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        if var is not None:
            v = d.variables.new()
            v.name = var[0]
            v.type = "SINGLE_PROP"
            v.targets[0].id_type = "OBJECT"
            v.targets[0].id = var[1]
            v.targets[0].data_path = var[2]
    return res


# ================================================================= per-instance kit
class Kit:
    """What a builder needs for one prop instance: names, colours, materials, objects, parameters.

        K = Kit(name, coll, root, slots)
        m, nt, out = K.new_mat("glaze")          material "<name>_glaze" with an empty node tree and an Output node
        o = K.to_obj("body", bm, [m], loc=...)   object "<name>_body" under the root
    """

    def __init__(self, name, coll, root, slots=None):
        self.name, self.coll, self.root = name, coll, root
        self.slots = dict(slots or {})
        self.colors = Colors(self.slots)
        sc = bpy.context.scene
        self.fps = sc.render.fps / sc.render.fps_base

    # ---- colours
    def slot(self, name, a=1.0):
        return self.colors.slot(name, a)

    def blend(self, **kw):
        return self.colors.blend(**kw)

    def light(self, **kw):
        return self.colors.light(**kw)

    # ---- names, materials, objects
    def oname(self, base):
        return f"{self.name}_{base}"

    def new_mat(self, base):
        m = bpy.data.materials.get(self.oname(base)) or bpy.data.materials.new(self.oname(base))
        m.use_nodes = True
        nt = m.node_tree
        clear_drivers(nt)
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.location = (900, 0)
        return m, nt, out

    def simple_mat(self, base, color, rough=0.6, metal=0.0, coat=0.0, spec=0.5, sheen=0.0, emit=None,
                   emit_strength=0.0):
        """Plain Principled material (color is a linear RGBA tuple, e.g. K.slot('rose'))."""
        m, nt, out = self.new_mat(base)
        b = principled(nt, out, **{"Base Color": color, "Roughness": rough, "Metallic": metal, "Coat Weight": coat,
                                   "Specular IOR Level": spec, "Sheen Weight": sheen})
        if emit is not None:
            b.inputs["Emission Color"].default_value = emit
            b.inputs["Emission Strength"].default_value = emit_strength
        return m

    def purge(self, name):
        o = bpy.data.objects.get(name)
        if o is not None:
            data = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if data is not None and data.users == 0:
                for coll in (bpy.data.meshes, bpy.data.curves, bpy.data.lights):
                    if coll.get(data.name) is data:
                        coll.remove(data)
                        break
        for coll in (bpy.data.meshes, bpy.data.curves):
            d = coll.get(name)
            if d is not None and d.users == 0:
                coll.remove(d)

    def obj(self, base, data, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), parent=None):
        """Object "<name>_<base>" with `data` (mesh, curve, None for an empty), linked to the prop's collection and
        parented to `parent` (default the root)."""
        name = self.oname(base)
        old = bpy.data.objects.get(name)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        o = bpy.data.objects.new(name, data)
        self.coll.objects.link(o)
        o.parent = parent if parent is not None else self.root
        o.location, o.rotation_euler, o.scale = loc, rot, scale
        return o

    def to_obj(self, base, bm, mats=(), loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), parent=None, keep_bm=False):
        """bmesh -> mesh object (materials appended in order = material_index), smooth flags as set on bm faces."""
        name = self.oname(base)
        self.purge(name)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        if not keep_bm:
            bm.free()
        for m in mats:
            me.materials.append(m)
        me.update()
        return self.obj(base, me, loc=loc, rot=rot, scale=scale, parent=parent)

    def empty(self, base, loc=(0, 0, 0), rot=(0, 0, 0), kind="PLAIN_AXES", size=0.02, parent=None, hidden=False):
        o = self.obj(base, None, loc=loc, rot=rot, parent=parent)
        o.empty_display_type = kind
        o.empty_display_size = size
        o.hide_render = hidden
        return o

    def load_image(self, path, colorspace="sRGB"):
        """Image datablock from a file (an existing one with the same path is reused and reloaded), or None when the file
        is missing."""
        import os
        path = os.path.expanduser(str(path))
        if not os.path.isfile(path):
            return None
        img = bpy.data.images.load(path, check_existing=True)
        img.reload()
        img.colorspace_settings.name = colorspace
        return img

    def collider(self, o):
        """Hidden from render, drawn as wire: a shape only the solvers and checks use."""
        as_collider(o)
        return o

    # ---- parameters
    def prop(self, name, value, lo=None, hi=None, doc=""):
        """Custom property on the root, with its UI range and description."""
        self.root[name] = value
        ui = {"description": doc, "default": value}
        if lo is not None:
            ui["min"] = lo
            ui["soft_min"] = lo
        if hi is not None:
            ui["max"] = hi
            ui["soft_max"] = hi
        self.root.id_properties_ui(name).update(**ui)

    def param(self, nt, prop, loc=(-2600, 600)):
        """Output socket of a Value node in `nt` driven by the root's custom property `prop`."""
        n = nt.nodes.new("ShaderNodeValue")
        n.name = n.label = f"mk_{prop}"
        n.location = loc
        n.outputs[0].default_value = float(self.root.get(prop, 0.0))       # the value before a driver evaluates
        drive(nt, f'nodes["{n.name}"].outputs[0].default_value', "p", var=("p", self.root, f'["{prop}"]'))
        return n.outputs[0]

    def scene_source(self, prop, marker="mk_set"):
        """The object that carries the scene-wide control `prop`: the root of a set marked with the custom property
        `marker` (sets/cafe.py marks its root) that has `prop`, else this prop's own root."""
        for o in bpy.data.objects:
            if o is not self.root and marker in o.keys() and prop in o.keys():
                return o
        return self.root

    def clock(self, nt, loc=(-2600, -700)):
        """Output socket of a Value node: the scene time in seconds (frame / fps)."""
        n = nt.nodes.new("ShaderNodeValue")
        n.name = n.label = "mk_clock"
        n.location = loc
        drive(nt, f'nodes["{n.name}"].outputs[0].default_value', f"frame / {self.fps:g}")
        return n.outputs[0]

    # ---- card
    def size(self, skip_colliders=True):
        """Bounding box [x, y, z] (m) of the prop's meshes, in the root's frame."""
        bpy.context.view_layer.update()
        inv = self.root.matrix_world.inverted()
        lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
        stack = list(self.root.children_recursive)
        for o in stack:
            if o.type not in ("MESH", "CURVE") or (skip_colliders and o.get("mk_collider")):
                continue
            M = inv @ o.matrix_world
            for c in o.bound_box:
                p = M @ Vector(c)
                lo = Vector(map(min, lo, p))
                hi = Vector(map(max, hi, p))
        return [round(v, 4) for v in (hi - lo)] if lo.x < 1e8 else [0.0, 0.0, 0.0]

    def card(self, use=None, colliders=None, origin="floor_center", front="-Y", size=None, **extra):
        return {"size": size or self.size(), "origin": origin, "front": front, "slots": self.colors.resolved(),
                "use": use or {}, "colliders": colliders or [], **extra}
