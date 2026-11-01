"""bake_lit: the bake's `lit` section (bake version 5), what mk's GPU renderer (mkr) needs to draw the lit frames Blender
renders, without Blender: the view transform captured as a lookup table, the film and EEVEE settings, the world, the
lights, every material's node graph, the images, the cameras' depth of field and every mesh the clip shows as Blender
renders it. bake.py makes a `Lit` before its frame loop, samples it on every frame of the loop, has it read the node
graphs and the meshes on the export frame (before the glTF export rewires the MMD materials) and writes it with the
rest; then `check` rebuilds the meshes from the files and measures them against Blender's own evaluation.

Files (in the bake beside bake.json): lit.bin (little-endian; BYTE offsets), view_lut.bin, images/ (packed and generated
images), and the per-frame values in bake.bin (float32; FLOAT offsets). Axes as in bake.bin: glTF's (Blender's
(x, y, z) is (x, z, -y)); what node graphs read (Generated, attributes) stays in Blender's.

bake.json "lit":
  file      "lit.bin"
  view      the colour management Blender's frames go through: transform, look, exposure, gamma, curve, display,
            dither (render dither_intensity), and `lut` {file "view_lut.bin", n, lo, hi}: float32 RGB at index
            ((b * n + g) * n + r) * 3, Blender's display-referred output (0..1, before 8-bit quantisation, no dither)
            for the scene-linear input 2^(lo + (hi - lo) * i / (n - 1)) per channel (i = r, g, b index), captured by
            saving a float image of those values through the scene's view settings as a 16-bit PNG; `zero` is the
            output for scene-linear 0 and `floor` the output for 2^lo
  film      filter_size, transparent; render: every RenderSettings value; eevee: every scene.eevee value
  world     null (no world: black, no light) or {name, use_nodes, color, props (every World value: probe_resolution,
            sun_threshold, sun_angle, use_sun_shadow, ...), visibility (its ray visibility flags: camera, diffuse,
            glossy, transmission, shadow, scatter), tree: GRAPH (null without nodes), anim}
  lights    [{name, type, shape, use_shadow, visible, aspect, props (every Light value), frames, offset}]: per frame
            (one when nothing changes) 22 floats at FLOAT offset + 22 k, in the world: position 3, forward 3 (the way it
            shines: local -Z), right 3 (local +X), up 3 (local +Y), colour 3 (linear), energy (W), size_x, size_y (area
            lights, m, the object's scale applied), spot_size (rad), spot_blend, radius (point, spot and area:
            shadow_soft_size as Blender uses it, not scaled; sun: its angle, rad), shown (1/0)
  materials {name: {settings (every Material value), tree: GRAPH (null without nodes), anim: [ANIM]}}
  groups    {name: {interface: [[in_out, name, identifier, socket_type, default]], tree: GRAPH, anim: [ANIM]}}
  images    {name: {file (absolute, or images/<file> inside the bake for packed and generated images), colorspace,
            alpha_mode, float, size, source, format, missing}}: `missing` is an image Blender cannot load, which it
            draws magenta (1, 0, 1, 1)
  cameras   {name: {type, offset}}: per frame 8 floats at FLOAT offset + 8 k: use_dof, focus_distance (m: the focus
            object's distance along the view axis when it has one), fstop, blades, rotation (rad), ratio, ortho_scale,
            sensor_width (mm). Lens, shift, matrix and clip range are bake.json's "cameras" (bake.py)
  objects   {name: {offset, frames}}: per frame the 3x4 world matrix of each object a node graph points at (a Texture
            Coordinate node's object)
  meshes    [MESH], every mesh the clip shows, as Blender renders it (by_frame ones with the render's modifier settings:
            `render_settings` {object: [what changed]})
  records   {name: {collections, props}}: the look records (bake.json `objects` has the others) of the objects lit meshes
            name as their `record` (an instance names its instancer) that no look sorts
  left_out  [{name, why, shows}]: what Blender renders that the bake cannot hold (a volume, hair, a mesh too big to
            bake on every frame, ...), with the clip frames [[k0, k1), ...] it shows on
  screen    null, or the screen layer: meshes [MESH by_frame of screen type (SCREEN below), with aspect, show [on, off)
            Blender frames, knockout], left_out [{name, why, aspect, show}] (a word the bake could not evaluate or hold:
            those frames' layers are Blender's), cameras {output: {type ORTHO, ortho_scale, sensor_fit, clip, size,
            matrix (3x4 glTF)}}, pass {engine, film_transparent, lights, world, motion_blur, dither, compositing,
            use_raytracing, samples {default, looks {shot or "medium <i>": {output: n}}}}, view and lut / zero / floor
            (as `view`, of its Standard transform: screen_lut.bin), format {file_format, color_mode, color_depth, alpha}
  points    {key: {expr, offset FLOAT (per frame x y z, world, glTF axes; NaN where no layer asks for it or it fails),
            frames [[k0, k1, shot], ...], errors [{error, frames}]}}: the layers' points (mkmmd.core.transition.demands);
            the point file is {p: [x, 1 - y], depth: z, m: hypot((qx - x) w / h, qy - y)} with (x, y, z) the point
            through the output's shot camera (world_to_camera_view) and (qx, qy) the point plus the camera's own +Y axis
  check     per mesh baked whole (static or skinned), its vertices rebuilt from the files on a few frames against
            Blender's evaluated mesh: the worst distance (mm) and the normals' worst and 99th-percentile angle (deg); per
            screen word (`screen`) its check frames decoded from the files against Blender's drawing of them (px in its
            output, normals' worst angle, attributes' worst difference) and its fit (frames, poses by kind, geometries,
            refits, projective maps, the worst error in px)

GRAPH {nodes: [NODE], links: [[from node, from socket identifier, to node, to socket identifier]]}: the valid, unmuted
  links between enabled sockets; node groups are not inlined. NODE {name, type, bl_idname, label, mute, parent, group (a
  group node's node group), image (an image node's image), props (every value of the node's own class: ID pointers by
  name, other pointer structs as nested props), ramp, curves, inputs and outputs [{identifier, name, type, enabled,
  linked, value}] (an output's value only for Value, RGB and Normal nodes), internal_links [[input, output]] (what a
  muted node passes through)}. ramp: {interpolation, color_mode, hue_interpolation, stops [[position, rgba]], table: 257
  rgba, ColorRamp.evaluate(i / 256)}. curves (RGB, vector and float curves): {extend, black_level, white_level, use_clip,
  clip [x0, y0, x1, y1], curves: [{name (R G B C, X Y Z or V), range [x0, x1], samples: 257 values of
  CurveMapping.evaluate at x0 + (x1 - x0) i / 256, slope_lo, slope_hi (per unit beyond the range), points}]}
ANIM {node, socket (identifier), io ("input" | "output"), size, offset, frames}: a socket value that is keyed or driven
  (a material morph's mmd_bind input, a keyed Value node), `size` floats per frame at FLOAT offset + size k
MESH {name, kind ("static" | "skinned" | "by_frame"), materials (per slot; null: an empty slot), visible {camera,
  shadow, diffuse, glossy, transmission, volume_scatter, holdout, shadow_catcher}, aspect (the only output it renders
  for, or null), object {color, pass_index}, object_attrs {"KIND:name": {offset FLOAT, frames}} (what its materials'
  Attribute nodes of type OBJECT, INSTANCER and VIEW_LAYER read off it, as Blender looks them up: a custom property,
  else an RNA path, of the object, else of its data (an instance: the instancer's particle settings and the instancer
  first; VIEW_LAYER: the view layer, the scene, the world); 4 floats per frame, a number as (v, v, v, 1), a vector of 1
  to 4 into (0, 0, 0, 1), none found (0, 0, 0, 0)), instance (null, or {by: instancer, of: source object, id}), shown
  (null: every frame, or FLOAT: per frame 1/0), matrix {offset FLOAT, frames}: per frame the 3x4 world matrix, column by
  column, armature (skinned: the bake.json "armatures" entry whose joints apply; a bone index is its "bones" order), dqs
  (false), vertices, attributes {key: {offset BYTE, comp, type ("f32" | "u16" | "u32")}}, triangles {offset BYTE, count}
  (u32, three vertex indices each, grouped by slot), slots [[slot, first triangle, count]], morphs (null, or {names,
  deltas BYTE (f32, key-major: key, vertex, xyz), values FLOAT (per frame len(names) floats)}), bend (below), by_frame
  (kind by_frame: {geoms: [{vertices, attributes, triangles, slots}] in the object's space, frames: [geom per frame, -1:
  hidden]}; its matrix places them on each frame, so an object that only moves holds one geometry)}
  Attribute keys: position (3: object space), normal (3: Blender's corner normals, custom ones too), generated (3:
  Blender's Generated coordinates, its axes), uv:<map> and uv: (the render-active map) (2), color:<name> and color: (the
  render colour attribute) (4, linear), attr:<name> (what an Attribute node of that name reads, comp by its type),
  tangent:<map> and tangent: (4: MikkTSpace tangent and bitangent sign), joints and weights (4 each), joints2 and
  weights2 (4 more: up to 8 bones), weights normalised as Blender's armature modifier does (all 0: not deformed), fan
  (u32) and bend (alpha, beta) for a mesh that bends. A vertex is split where Blender's corner data differs, and in a
  mesh that bends where its corner fan does. A vertex v (position plus sum_k values_k deltas_k) is at M v on a frame (M
  its matrix), a skinned one at sum_i w_i J_i M v (J the armature's joints on that frame).
  Normals: a static mesh's turn with its matrix. A mesh that bends (skinned or shape-keyed) has bend {fans, center, ref,
  other (BYTE u32 per fan: a vertex at the fan's own point, at the far end of its reference edge, of its other edge),
  corners, edges, faces (each {count, offsets BYTE u32 [count + 1], items BYTE u32, size}: per fan its corners as
  (face, previous vertex, next vertex) triples, per fan its edges' far-end vertices, per face its triangles)}, and its
  normals are rebuilt on every frame from the posed vertices P as Blender rebuilds them (mkmmd/core/bend.py: `normals`
  is the reference): per face n_f = the unit sum of its triangles' cross products; per fan lnor = unit sum over its
  corners of n_f times the face's angle at the corner, ra = the mean angle between lnor and its edges, ref = the
  reference edge flattened onto lnor's plane, ortho = lnor x ref, span = the angle round lnor from ref to the other
  edge flattened alike (a whole turn when other is ref's vertex); per vertex a = alpha ra (alpha < 0: 2 pi + alpha
  (2 pi - ra)), b = beta span, normal = cos(a) lnor + sin(a) (cos(b) ref + sin(b) ortho).
SCREEN a word of screen type (mkmmd/core/fitgeo.py makes it, mkr/mkr/src/screen_geo.rs decodes it): MESH by_frame
  whose by_frame is {geoms [FIT], frames [geom per frame, -1: hidden], poses [BYTE per frame: its POSE, -1: hidden]}, in
  the world (the plane z = 0, a unit a frame height: glTF x and z are the plane, y its depth). Its corners are one vertex
  where their normals round alike at 1 / 64 (a flat word's all share its plane's) and its other corner data agree.
  FIT {vertices, triangles {offset (an id: the island table's BYTE offset), count}, slots: the frame it makes (below),
  attributes {}, fit {protos {count, table BYTE (u32 x 4 per prototype: first vertex, vertices, first triangle,
  triangles), vertices, attributes {key: {offset BYTE, comp, type f32}} (per prototype vertex: position and normal, the
  first island of the shape as the geometry's first frame has it, and the attributes that vary over an island),
  triangles {offset BYTE, count} (u32, three per triangle, each its prototype's own vertex numbers)}, islands {count,
  table BYTE (u32 x 4 per island: prototype, slot, first vertex, first triangle)}, attrs [[key, comp]] (the attributes
  one value per island holds: in every POSE)}}. An island is connected triangles of one slot (joined where they share
  one of Blender's vertices); islands alike in shape, triangles and per-vertex attributes share a prototype.
  The frame: island after island, its prototype's vertices through the island's map (from its first vertex on) and its
  prototype's triangles plus its first vertex (from its first triangle on: slots group them); attributes per vertex
  from the prototype, an island attribute its island's value. POSE (16-byte aligned): u32 kind (0: a map per island; 1:
  one map of the whole word, after each island's map in the pose of bake frame `ref`), u32 attrs (1: values follow; 0:
  those of the pose it follows), u32 ref, u32 0; then f32 maps (kind 0: islands x 16, kind 1: 16), each a 4x4 M
  row-major on q = (x, y, z, 1): x' = (M0 . q) / (M3 . q), y' = M1 . q, z' = (M2 . q) / (M3 . q) (affine: M3 = (0, 0,
  0, 1); a tilt seen in perspective keeps each point's depth y); then with attrs, per attrs entry islands x comp f32.
  Normals: the prototype's turned by the cofactor det(J) J^-T of each map's Jacobian J at the point, then normalised.
  Every frame's maps are fitted in float64 and measured as decoded from float32 (mkmmd.core.fitgeo.TOL: 1e-5 frame
  heights; normals within NORMAL_TOL); a frame that misses starts a new geometry, its own first frame. A word's
  encoding (its part of lit.bin, record, materials, stats and check samples) is cached in <project>/.mk/cache/screen/
  by `_screen_key` (this file's and fitgeo.py's source are part of the key); a word found there is not evaluated.

Self-check: `check` reads lit.bin and bake.bin back and rebuilds every static and skinned mesh on the export frame, the
middle frame and the last (positions and normals), against the evaluated mesh Blender had on those frames; and every
screen word on the first, middle and last frame of its window against Blender's drawing of them."""
import functools
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
import zlib

import bpy
import numpy as np

from ...core import bend as BD
from ...core import fitgeo as FG
from ...core import screentype as SR
from ...core import transition as TR
from ... import cache as CA
from .. import styles as ST
from ..ops_core import _namespace

AXES3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
AXES4 = np.eye(4)
AXES4[:3, :3] = AXES3
LUT_N, LUT_LO, LUT_HI = 65, -12.5, 12.5
LIGHT_FLOATS, CAMERA_FLOATS = 22, 8
FLEX_CAP = 256_000_000                  # bytes one object baked frame by frame may add to lit.bin
CHECK_MM = 0.1                          # a mesh rebuilt from the bake further than this from Blender's is reported
SCREEN_CAP = 512_000_000                # bytes the screen layer's words may add to lit.bin, all of them (fitted)
NORMAL_KEY = 64                         # screen type: corners whose normals round alike at 1/64 are one vertex
SCREEN_CHECK = 2048                     # vertices of a screen word kept on each of its check frames
MAX_BONES = 8                           # bone weights a skinned vertex may have (joints + joints2)
IMAGE_NODES = ("TEX_IMAGE", "TEX_ENVIRONMENT")
OBJECT_ATTRS = ("OBJECT", "INSTANCER", "VIEW_LAYER")     # Attribute node types read per object, not per vertex
OUT_VALUES = ("VALUE", "RGB", "NORMAL")    # nodes whose output socket holds their value
EVALUATED = ("CURVE", "SURFACE", "FONT", "META")
UNDRAWN = {"CURVES": "a hair curves object", "POINTCLOUD": "a point cloud", "VOLUME": "a volume",
           "GPENCIL": "grease pencil", "GREASEPENCIL": "grease pencil"}
PROCEDURAL = ("TEX_NOISE", "TEX_VORONOI", "TEX_WAVE", "TEX_GRADIENT", "TEX_MAGIC", "TEX_BRICK", "TEX_CHECKER",
              "TEX_WHITE_NOISE", "TEX_MUSGRAVE", "TEX_GABOR")       # an unlinked Vector reads Generated
ATTR_TYPES = {"FLOAT": ("value", 1, np.float32), "INT": ("value", 1, np.int32), "INT8": ("value", 1, np.int8),
              "BOOLEAN": ("value", 1, bool), "FLOAT2": ("vector", 2, np.float32), "INT32_2D": ("value", 2, np.int32),
              "FLOAT_VECTOR": ("vector", 3, np.float32), "FLOAT_COLOR": ("color", 4, np.float32),
              "BYTE_COLOR": ("color", 4, np.float32), "QUATERNION": ("value", 4, np.float32)}
CURVE_NAMES = {"CURVE_RGB": "RGBC", "CURVE_VEC": "XYZ", "CURVE_FLOAT": "V"}
DTYPES = {"f32": "<f4", "u16": "<u2", "u32": "<u4", "u8": "u1"}
_PATH = re.compile(r'^(?:node_tree\.)?nodes\["((?:[^"\\]|\\.)*)"\]\.(inputs|outputs)\[(\d+|"(?:[^"\\]|\\.)*")\]'
                   r'\.default_value$')


def _gl4(m):
    """Blender matrices (..., 4, 4) in glTF's axes."""
    return AXES4 @ np.asarray(m, float) @ AXES4.T


def _affine(m):
    """(..., 4, 4) -> (..., 12): the top three rows, column by column."""
    m = np.asarray(m, float)
    return np.swapaxes(m[..., :3, :], -1, -2).reshape(*m.shape[:-2], 12)


def _get(coll, prop, n, comp, dtype=np.float32):
    a = np.empty(n * comp, dtype)
    coll.foreach_get(prop, a)
    return a.reshape(n, comp) if comp > 1 else a


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


def _ranges(flags):
    """[[k0, k1), ...] where `flags` is true."""
    out, start = [], None
    for k, v in enumerate(flags):
        if v and start is None:
            start = k
        elif not v and start is not None:
            out.append([start, k])
            start = None
    if start is not None:
        out.append([start, len(flags)])
    return out


# ================================================================================================= values and graphs
def _val(v):
    """An RNA value as JSON."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, bpy.types.ID):
        return v.name
    try:
        return [_val(x) for x in v]
    except TypeError:
        return str(v)


_SKIP = {}


def _skip(kind):
    """Property identifiers left out of a record: the ID's own (name, users, ...) or a node's base class's."""
    if kind not in _SKIP:
        types = {"id": (bpy.types.ID,), "node": (bpy.types.Node, bpy.types.NodeInternal, bpy.types.ShaderNode)}[kind]
        _SKIP[kind] = {p.identifier for t in types for p in t.bl_rna.properties} | {"animation_data", "rna_type"}
    return _SKIP[kind]


def _props(obj, skip=(), depth=1):
    """Every non-collection RNA value of `obj` by identifier: ID pointers by name, other pointer structs nested (to
    `depth`), colour ramps, curve mappings and nodes left out (they have records of their own)."""
    out = {}
    for p in obj.bl_rna.properties:
        k = p.identifier
        if k in skip or k == "rna_type" or p.type == "COLLECTION":
            continue
        try:
            v = getattr(obj, k)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        if p.type == "POINTER":
            if v is None or isinstance(v, bpy.types.ID):
                out[k] = None if v is None else v.name
            elif depth > 0 and not isinstance(v, (bpy.types.ColorRamp, bpy.types.CurveMapping, bpy.types.Node,
                                                   bpy.types.NodeSocket, bpy.types.NodeTree, bpy.types.AnimData)):
                out[k] = _props(v, depth=depth - 1)
            continue
        out[k] = _val(v)
    return out


def _ramp(cr):
    return {"interpolation": cr.interpolation, "color_mode": cr.color_mode, "hue_interpolation": cr.hue_interpolation,
            "stops": [[e.position, list(e.color)] for e in cr.elements],
            "table": [list(cr.evaluate(i / 256.0)) for i in range(257)]}


def _curves(cm, kind):
    """A curve mapping as samples of Blender's own evaluation over the range it builds its tables on (the clip range)."""
    cm.initialize()
    x0, x1 = cm.clip_min_x, cm.clip_max_x
    names = CURVE_NAMES.get(kind, "")
    xs = x0 + (x1 - x0) * np.arange(257) / 256.0
    out = {"extend": cm.extend, "black_level": list(cm.black_level), "white_level": list(cm.white_level),
           "use_clip": cm.use_clip, "clip": [cm.clip_min_x, cm.clip_min_y, cm.clip_max_x, cm.clip_max_y], "curves": []}
    for i, c in enumerate(cm.curves):
        ys = [cm.evaluate(c, float(x)) for x in xs]
        out["curves"].append({"name": names[i] if i < len(names) else str(i), "range": [x0, x1], "samples": ys,
                              "slope_lo": ys[0] - cm.evaluate(c, x0 - 1.0), "slope_hi": cm.evaluate(c, x1 + 1.0) - ys[-1],
                              "points": [[p.location[0], p.location[1], p.handle_type] for p in c.points]})
    return out


def _socket(s, linked, value=True):
    return {"identifier": s.identifier, "name": s.name, "type": s.type, "enabled": bool(s.enabled), "linked": linked,
            "value": _val(s.default_value) if value and hasattr(s, "default_value") else None}


def graph(nt, used):
    """GRAPH of the node tree `nt` (module docstring); adds the node groups, images and objects it names to `used`."""
    links = [[ln.from_node.name, ln.from_socket.identifier, ln.to_node.name, ln.to_socket.identifier] for ln in nt.links
             if ln.is_valid and not ln.is_muted and ln.from_socket.enabled and ln.to_socket.enabled]
    into, out_of = {(a[2], a[3]) for a in links}, {(a[0], a[1]) for a in links}
    nodes = []
    for n in nt.nodes:
        rec = {"name": n.name, "type": n.type, "bl_idname": n.bl_idname, "label": n.label, "mute": n.mute,
               "parent": n.parent.name if n.parent else None, "group": None, "image": None,
               "props": _props(n, skip=_skip("node")), "ramp": None, "curves": None,
               "inputs": [_socket(s, (n.name, s.identifier) in into) for s in n.inputs],
               "outputs": [_socket(s, (n.name, s.identifier) in out_of, n.type in OUT_VALUES) for s in n.outputs],
               "internal_links": [[ln.from_socket.identifier, ln.to_socket.identifier] for ln in n.internal_links]}
        if n.type == "GROUP" and n.node_tree is not None:
            rec["group"] = n.node_tree.name
            used["groups"].add(n.node_tree.name)
        if n.type in IMAGE_NODES and getattr(n, "image", None) is not None:
            rec["image"] = n.image.name
            used["images"].add(n.image.name)
        if isinstance(getattr(n, "object", None), bpy.types.Object):
            used["objects"].add(n.object.name)
        if isinstance(getattr(n, "color_ramp", None), bpy.types.ColorRamp):
            rec["ramp"] = _ramp(n.color_ramp)
        if isinstance(getattr(n, "mapping", None), bpy.types.CurveMapping):
            rec["curves"] = _curves(n.mapping, n.type)
        nodes.append(rec)
    return {"nodes": nodes, "links": links}


def _interface(ng):
    return [[it.in_out, it.name, it.identifier, it.socket_type, _val(getattr(it, "default_value", None))]
            for it in ng.interface.items_tree if it.item_type == "SOCKET"]


def _anim_paths(*owners):
    """The data paths keyed (actions, NLA strips) or driven on these IDs."""
    out = set()
    for o in owners:
        ad = getattr(o, "animation_data", None) if o is not None else None
        if ad is None:
            continue
        for act in [ad.action] + [s.action for t in ad.nla_tracks for s in t.strips]:
            if act is not None:
                out.update(fc.data_path for fc in act.fcurves)
        out.update(d.data_path for d in ad.drivers)
    return out


def _channels(nt, owner=None):
    """[(node name, socket, "input" | "output")] of the sockets of `nt` whose value is keyed or driven, on the tree or
    on its owner (a material keys its tree's nodes as node_tree.nodes[...])."""
    out = []
    for path in sorted(_anim_paths(nt, owner)):
        m = _PATH.match(path)
        if m is None:
            continue
        node = nt.nodes.get(bpy.utils.unescape_identifier(m.group(1)))
        if node is None:
            continue
        socks = node.inputs if m.group(2) == "inputs" else node.outputs
        key = m.group(3)
        if key.isdigit():
            s = socks[int(key)] if int(key) < len(socks) else None
        else:
            s = socks.get(bpy.utils.unescape_identifier(key[1:-1]))
        if s is not None and hasattr(s, "default_value"):
            out.append((node.name, s, "input" if m.group(2) == "inputs" else "output"))
    return out


def _reads(trees):
    """What the node trees read off a mesh, as attribute keys (module docstring): "generated", "uv:<map>", "color:<name>",
    "attr:<name>", "tangent:<map>" (an empty name is the render-active one); node groups followed. Trees are told apart
    by identity: every material's own tree is named "Shader Nodetree"."""
    out, seen, todo = set(), set(), list(trees)
    while todo:
        nt = todo.pop()
        if nt is None or nt.as_pointer() in seen:
            continue
        seen.add(nt.as_pointer())
        for n in nt.nodes:
            t = n.type
            vec = n.inputs.get("Vector") if t in IMAGE_NODES + PROCEDURAL else None
            if t == "TEX_IMAGE" and vec is not None and not vec.is_linked:
                out.add("uv:")
            elif t in PROCEDURAL and vec is not None and not vec.is_linked:
                out.add("generated")
            elif t == "TEX_COORD":
                for s in n.outputs:
                    if s.is_linked and s.name in ("UV", "Generated"):
                        out.add("uv:" if s.name == "UV" else "generated")
            elif t == "UVMAP":
                out.add("uv:" + n.uv_map)
            elif t == "ATTRIBUTE" and n.attribute_type == "GEOMETRY":
                out.add("attr:" + n.attribute_name)
            elif t == "VERTEX_COLOR":
                out.add("color:" + n.layer_name)
            elif t == "NORMAL_MAP" and n.space == "TANGENT":
                out.add("tangent:" + n.uv_map)
            elif t == "TANGENT":
                out.add("tangent:" + n.uv_map if n.direction_type == "UV_MAP" else "generated")
            elif t == "NEW_GEOMETRY" and n.outputs["Tangent"].is_linked:
                out.add("generated")
            elif t == "GROUP" and n.node_tree is not None:
                todo.append(n.node_tree)
    return out


def _object_reads(trees):
    """What the node trees' Attribute nodes of type OBJECT, INSTANCER and VIEW_LAYER read, as sorted "KIND:name" keys
    (module docstring: object_attrs); node groups followed, trees told apart by identity (as `_reads`)."""
    out, seen, todo = set(), set(), list(trees)
    while todo:
        nt = todo.pop()
        if nt is None or nt.as_pointer() in seen:
            continue
        seen.add(nt.as_pointer())
        for n in nt.nodes:
            if n.type == "ATTRIBUTE" and n.attribute_type in OBJECT_ATTRS:
                out.add(f"{n.attribute_type}:{n.attribute_name}")
            elif n.type == "GROUP" and n.node_tree is not None:
                todo.append(n.node_tree)
    return sorted(out)


def _rgba(v):
    """A property's value as an Attribute node reads it: a number (int, float, bool) as (v, v, v, 1), a vector of 1 to 4
    numbers into (0, 0, 0, 1); None for any other value (a string, a group, a matrix, a pointer)."""
    if isinstance(v, (bool, int, float)):
        return (float(v),) * 3 + (1.0,)
    if isinstance(v, (str, bytes)) or hasattr(v, "keys"):
        return None
    try:
        vals = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return tuple(vals + [0.0, 0.0, 0.0, 1.0][len(vals):]) if 1 <= len(vals) <= 4 else None


def _lookup(owner, name):
    """`name` on a data-block (or view layer) as an Attribute node looks it up: a custom property first, else an RNA
    path; its RGBA, or None."""
    if owner is None:
        return None
    try:
        if name in owner.keys():
            v = _rgba(owner[name])
            if v is not None:
                return v
    except TypeError:                                  # no custom properties
        pass
    try:
        return _rgba(owner.path_resolve(name))
    except (ValueError, TypeError):
        return None


def _object_attr(key, ob, sc, by=None, psys=None):
    """The RGBA an Attribute node reads for "KIND:name" `key` on object `ob` (an instance: of instancer `by`, made by
    particle system `psys` or None) in scene `sc`, (0, 0, 0, 0) when nothing has it (module docstring: object_attrs)."""
    kind, name = key.split(":", 1)
    if kind == "VIEW_LAYER":
        chain = (bpy.context.view_layer, sc, sc.world)
    else:
        chain = (ob, ob.data)
        if kind == "INSTANCER" and by is not None:              # its particle settings, then the instancer object
            chain = (psys.settings if psys is not None else None, by) + chain
    for owner in chain:
        v = _lookup(owner, name)
        if v is not None:
            return v
    return (0.0, 0.0, 0.0, 0.0)


# ================================================================================================= colour management
def view(sc, part):
    """The view record (module docstring), its LUT written to part/view_lut.bin."""
    ims = sc.render.image_settings
    override = getattr(ims, "color_management", "FOLLOW_SCENE") == "OVERRIDE"
    vs, ds = (ims.view_settings, ims.display_settings) if override else (sc.view_settings, sc.display_settings)
    lut, zero, floor = _capture(sc, part)
    lut.astype("<f4").tofile(os.path.join(part, "view_lut.bin"))
    return {"transform": vs.view_transform, "look": vs.look, "exposure": vs.exposure, "gamma": vs.gamma,
            "curve": vs.use_curve_mapping, "display": ds.display_device, "dither": sc.render.dither_intensity,
            "override": override, "lut": {"file": "view_lut.bin", "n": LUT_N, "lo": LUT_LO, "hi": LUT_HI},
            "zero": zero.tolist(), "floor": floor.tolist()}


def _capture(sc, part):
    """Blender's display transform of the LUT grid, 0 and 2^lo: a float image of those scene-linear values saved through
    the scene's view settings as a 16-bit PNG (no dither) and read back as it is. (n^3, 3), (3,), (3,)."""
    n = LUT_N
    v = 2.0 ** (LUT_LO + (LUT_HI - LUT_LO) * np.arange(n) / (n - 1))
    b, g, r = np.meshgrid(v, v, v, indexing="ij")              # raveled, r runs fastest: ((b * n + g) * n + r)
    px = np.concatenate([np.stack([r.ravel(), g.ravel(), b.ravel()], 1), [[0.0, 0.0, 0.0], [v[0]] * 3]])
    w = n * n
    h = -(-len(px) // w)
    buf = np.zeros((h * w, 4), np.float32)
    buf[:len(px), :3] = px
    buf[:, 3] = 1.0
    path = os.path.join(part, "view_lut.png")
    ims, r_ = sc.render.image_settings, sc.render
    saved = [(ims, k, getattr(ims, k)) for k in ("file_format", "color_mode", "color_depth")]
    saved.append((r_, "dither_intensity", r_.dither_intensity))
    img = bpy.data.images.new("mk_view_lut", w, h, alpha=True, float_buffer=True)
    try:
        ims.file_format = "PNG"
        ims.color_mode, ims.color_depth = "RGB", "16"
        r_.dither_intensity = 0.0
        img.pixels.foreach_set(buf.ravel())
        img.save_render(path, scene=sc)
    finally:
        for owner, k, val in reversed(saved):
            setattr(owner, k, val)
        bpy.data.images.remove(img)
    back = bpy.data.images.load(path, check_existing=False)
    try:
        back.colorspace_settings.name = "Non-Color"                # the file's values as they are
        out = np.empty(back.size[0] * back.size[1] * 4, np.float32)
        back.pixels.foreach_get(out)
    finally:
        bpy.data.images.remove(back)
        os.remove(path)
    out = out.reshape(-1, 4)[:len(px), :3]
    return out[:n ** 3], out[n ** 3], out[n ** 3 + 1]


# ================================================================================================= images
def _png(path, px, w, h, bits):
    """Write (h * w * 4) floats 0..1 (Blender's rows, bottom first) as an RGBA PNG of `bits` per channel, as they are."""
    a = np.clip(px.reshape(h, w, 4)[::-1], 0.0, 1.0)
    rows = np.round(a * 255).astype(np.uint8) if bits == 8 else np.round(a * 65535).astype(">u2")
    raw = b"".join(b"\0" + row.tobytes() for row in rows)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, 6, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _image(im, part, i):
    """An image's record (module docstring); a packed or generated one is written into part/images/."""
    rec = {"file": None, "colorspace": im.colorspace_settings.name, "alpha_mode": im.alpha_mode, "float": bool(im.is_float),
           "size": [0, 0], "source": im.source, "format": im.file_format, "missing": False}
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", im.name).strip("._") or "image"
    if im.packed_file is not None:
        ext = os.path.splitext(im.filepath or "")[1] or "." + im.file_format.lower().replace("jpeg", "jpg")
        rel = f"images/{i:03d}_{os.path.splitext(safe)[0]}{ext}"
        os.makedirs(os.path.join(part, "images"), exist_ok=True)
        with open(os.path.join(part, rel), "wb") as fh:
            fh.write(im.packed_file.data)
        rec["file"] = rel
    elif im.source == "GENERATED":
        w, h = im.size
        rel = f"images/{i:03d}_{os.path.splitext(safe)[0]}.png"
        os.makedirs(os.path.join(part, "images"), exist_ok=True)
        px = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(px)
        _png(os.path.join(part, rel), px, w, h, 16 if im.is_float else 8)
        rec["file"] = rel
    elif im.source in ("FILE", "SEQUENCE", "MOVIE", "TILED"):
        rec["file"] = os.path.normpath(bpy.path.abspath(im.filepath, library=im.library)) if im.filepath else None
    try:
        rec["size"] = [int(x) for x in im.size]                     # loads it
    except RuntimeError:
        pass
    local = rec["file"] is not None and (rec["file"].startswith("images/") or os.path.isfile(rec["file"]))
    rec["missing"] = not local or 0 in rec["size"]
    return rec


# ================================================================================================= meshes
def _attr_corners(a, cv, cpoly):
    """An attribute's values per corner, float32 (n corners, comp), or None (a domain or type a render cannot read)."""
    t = ATTR_TYPES.get(a.data_type)
    if t is None or a.domain not in ("POINT", "CORNER", "FACE"):
        return None
    prop, comp, dt = t
    arr = np.empty(len(a.data) * comp, dt)
    a.data.foreach_get(prop, arr)
    arr = arr.reshape(-1, comp).astype(np.float32)
    return arr[cv] if a.domain == "POINT" else arr[cpoly] if a.domain == "FACE" else arr


def _tangents(me, uv):
    """MikkTSpace tangents of UV map `uv` per corner (n corners, 4: xyz, bitangent sign), or None."""
    try:
        me.calc_tangents(uvmap=uv)
    except RuntimeError:                       # faces of more than four corners
        return None
    n = len(me.loops)
    t = np.concatenate([_get(me.loops, "tangent", n, 3), _get(me.loops, "bitangent_sign", n, 1)[:, None]], 1)
    me.free_tangents()
    return t


def _corner_data(me, reads, cv, cpoly):
    """The data each attribute key in `reads` names that `me` has, per corner: ({source: array}, {key: source}). A
    source is an attribute's name (UV maps and colour attributes are attributes) or "tangent:<map>"."""
    uvs = [u.name for u in me.uv_layers]
    act_uv = next((u.name for u in me.uv_layers if u.active_render), uvs[0] if uvs else None)
    cols = me.color_attributes
    ri = getattr(cols, "render_color_index", -1)
    act_col = cols[ri].name if 0 <= ri < len(cols) else None
    data, alias = {}, {}
    for key in sorted(reads):
        kind, _, name = key.partition(":")
        if kind == "uv":
            src = name or act_uv
            ok = src in uvs
        elif kind == "color":
            src = name or act_col
            ok = src is not None and cols.get(src) is not None
        elif kind == "attr":
            src = name
            ok = bool(name) and not name.startswith(".") and me.attributes.get(name) is not None
        elif kind == "tangent":
            src = "tangent:" + (name or act_uv or "")
            ok = (name or act_uv) in uvs
        else:
            continue
        if not ok:
            continue
        if src not in data:
            arr = _tangents(me, src[8:]) if kind == "tangent" else _attr_corners(me.attributes[src], cv, cpoly)
            if arr is None:
                continue
            data[src] = arr
        alias[key] = src
    return data, alias


def _split(cv, cols, ints=()):
    """Corners of one vertex with the same data (float columns `cols`, integer columns `ints`) are one vertex: (each
    vertex's first corner, each corner's vertex), vertices in the order of Blender's, then of their first corner."""
    parts = [cv.astype(np.uint32)[:, None]] + [np.asarray(i).astype(np.uint32).reshape(len(cv), -1) for i in ints]
    for c in cols:
        c = np.ascontiguousarray(c, np.float32).reshape(len(cv), -1) + np.float32(0.0)     # -0.0 is 0.0
        parts.append(c.view(np.uint32))
    m = np.ascontiguousarray(np.concatenate(parts, 1))
    key = m.view(np.dtype((np.void, 4 * m.shape[1]))).ravel()
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    order = np.lexsort((first, cv[first]))
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    return first[order], rank[inv.ravel()]


def _geom(me, reads, nslots, world=None, base=None, orco=None, texspace=None, generated=True, fans=None, merge=False):
    """A Blender mesh as drawable arrays: {src (Blender's vertex of each vertex), corner (each corner's vertex), attrs
    {position, normal[, generated]} and named {source: array} with alias {key: source}, tris (m, 3) uint32 grouped by
    slot, tri_loops (each of those triangles' corners), slots [[slot, first, count]], first (each vertex's first
    corner)}, or None (nothing to draw).
    `world` (4x4, Blender's axes) puts positions, normals and tangents in the world, else they stay in object space;
    `base`: the positions per Blender vertex (the reference shape key's), `orco` and `texspace` the points and texture
    space Generated is made of (default: the mesh's); `fans`: a fan per corner, never two in one vertex; `merge`:
    corners whose normals agree to within NORMAL_KEY are one vertex whatever the normals' domain (screen type: the
    corners of a flat word all have its plane's normal, but for float32 noise on sliver faces)."""
    nv, nl = len(me.vertices), len(me.loops)
    if not nl or not len(me.polygons):
        return None
    if world is not None and abs(np.linalg.det(np.asarray(world, float)[:3, :3])) < 1e-12:
        return None                                        # scaled to nothing: it shows nothing
    co = _get(me.vertices, "co", nv, 3)
    cv = _get(me.loops, "vertex_index", nl, 1, np.int32)
    tot = _get(me.polygons, "loop_total", len(me.polygons), 1, np.int32)
    cpoly = np.repeat(np.arange(len(tot), dtype=np.int32), tot)
    cn = _get(me.corner_normals, "vector", nl, 3)
    data, alias = _corner_data(me, reads, cv, cpoly)
    srcs = sorted(data)
    cols = [data[s] for s in srcs]
    dom = me.normals_domain                                # vertex normals split nothing, face normals split faces
    if merge and dom != "POINT":
        first, corner = _split(cv, [_normal_key(cn)] + cols, [fans] if fans is not None else [])
    else:
        ints = ([cpoly] if dom == "FACE" else []) + ([fans] if fans is not None else [])
        first, corner = _split(cv, cols if dom != "CORNER" else [cn] + cols, ints)
    src = cv[first]
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tl = _get(me.loop_triangles, "loops", nt, 3, np.int32)
    mi = np.clip(_get(me.loop_triangles, "material_index", nt, 1, np.int32), 0, max(nslots - 1, 0))
    order = np.argsort(mi, kind="stable")
    tris = corner[tl[order]].astype(np.uint32)
    slot, start, count = np.unique(mi[order], return_index=True, return_counts=True)
    pos = (co if base is None else base)[src].astype(float)
    nor = cn[first].astype(float)
    named = {s: data[s][first] for s in srcs}
    if world is not None:
        W = np.asarray(world, float)
        pos = pos @ W[:3, :3].T + W[:3, 3]
        nor = nor @ np.linalg.inv(W[:3, :3])
        for s in srcs:
            if s.startswith("tangent:"):
                named[s][:, :3] = named[s][:, :3] @ W[:3, :3].T
    nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)
    for s in srcs:
        if s.startswith("tangent:"):
            t = named[s][:, :3].astype(float)
            named[s][:, :3] = (t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)) @ AXES3.T
    attrs = {"position": (pos @ AXES3.T).astype(np.float32), "normal": (nor @ AXES3.T).astype(np.float32)}
    if generated:
        oc = co if orco is None else orco
        loc, size = texspace if texspace is not None else (np.array(me.texspace_location), np.array(me.texspace_size))
        size = np.where(np.abs(size) > 0.0, size, 1.0)
        attrs["generated"] = (((oc[src] - loc) / size) * 0.5 + 0.5).astype(np.float32)
    return {"src": src, "corner": corner, "first": first, "attrs": attrs, "named": named, "alias": alias, "tris": tris,
            "tri_loops": tl[order], "slots": [[int(s), int(a), int(c)] for s, a, c in zip(slot, start, count)]}


def _normal_key(cn):
    """Corner normals rounded to 1 / NORMAL_KEY: corners whose keys agree are one vertex when `_geom` merges."""
    return np.round(np.asarray(cn, np.float32) * NORMAL_KEY).astype(np.float32)


def _corner_verts(me):
    """Each corner's vertex, off the mesh's own array (the loops collection reads it 300 times slower)."""
    a = me.attributes.get(".corner_vert")
    if a is not None and a.domain == "CORNER" and len(a.data) == len(me.loops):
        return _get(a.data, "value", len(me.loops), 1, np.int32)
    return _get(me.loops, "vertex_index", len(me.loops), 1, np.int32)


def _face_materials(me):
    """Each face's material index (0 without the attribute), off its array."""
    a = me.attributes.get("material_index")
    if a is None or a.domain != "FACE":
        return np.zeros(len(me.polygons), np.int32)
    return _get(a.data, "value", len(me.polygons), 1, np.int32)


def _loop_totals(me):
    """Each face's corner count (a mesh of triangles needs no reading)."""
    if len(me.loops) == 3 * len(me.polygons):
        return np.full(len(me.polygons), 3, np.int32)
    return _get(me.polygons, "loop_total", len(me.polygons), 1, np.int32)


def _corner_normals(me, cv, cpoly):
    """me.corner_normals, read off the domain Blender keeps them on (a face's and a vertex's are one value each)."""
    dom = me.normals_domain
    if dom == "FACE":
        return _get(me.polygon_normals, "vector", len(me.polygons), 3)[cpoly]
    if dom == "POINT":
        return _get(me.vertex_normals, "vector", len(me.vertices), 3)[cv]
    return _get(me.corner_normals, "vector", len(me.loops), 3)


def _write_geom(blob, g):
    """Write a _geom's arrays to lit.bin: (attributes {key: spec}, triangles spec)."""
    attrs = {k: {"offset": blob.put(a, "f32"), "comp": int(a.shape[1]), "type": "f32"} for k, a in g["attrs"].items()}
    at = {s: {"offset": blob.put(a, "f32"), "comp": int(a.shape[1]), "type": "f32"} for s, a in g["named"].items()}
    attrs.update({k: at[s] for k, s in g["alias"].items()})
    return attrs, {"offset": blob.put(g["tris"], "u32"), "count": int(len(g["tris"]))}


# ================================================================================================= normals that bend
def _fans(me, cv, cpoly):
    """Blender's corner fans of a mesh (mkmmd.core.bend.fans): (fan of each corner, how many, smooth per edge, each
    corner's next and previous corner in its face, each corner's edge)."""
    nl, npoly, ne = len(me.loops), len(me.polygons), len(me.edges)
    ce = _get(me.loops, "edge_index", nl, 1, np.int32)
    sharp = me.attributes.get("sharp_edge")
    sharp = _get(sharp.data, "value", ne, 1, bool) if sharp is not None and sharp.domain == "EDGE" else np.zeros(ne, bool)
    fan, nfan, smooth_e, cnext, cprev = BD.fans(cv, cpoly, ce, _get(me.polygons, "loop_start", npoly, 1, np.int32),
                                                _get(me.polygons, "loop_total", npoly, 1, np.int32), sharp,
                                                _get(me.polygons, "use_smooth", npoly, 1, bool))
    return fan, nfan, smooth_e, cnext, cprev, ce


def _pairs(me):
    """Every (vertex, vertex group, weight) of the mesh as three arrays."""
    flat = np.array([(v.index, g.group, g.weight) for v in me.vertices for g in v.groups], float).reshape(-1, 3)
    return flat[:, 0].astype(np.int64), flat[:, 1].astype(np.int64), flat[:, 2]


def _skin(ob, me, arm, pairs):
    """Per Blender vertex its bones (indices in the armature's pose bones) and weights, normalised over its deforming
    bones as the armature modifier does (none: all 0), 4 or 8 of each: (joints (nv, k) uint16, weights (nv, k)), or
    (None, why)."""
    bones = {b.name: i for i, b in enumerate(arm.pose.bones)}
    deform = {b.name for b in arm.data.bones if b.use_deform}
    gmap = np.full(max(len(ob.vertex_groups), 1), -1, np.int64)
    for g in ob.vertex_groups:
        if g.name in deform:
            gmap[g.index] = bones[g.name]
    vi, gi, w = pairs
    b = gmap[gi] if len(gi) else gi
    ok = (b >= 0) & (w > 0.0)
    vi, b, w = vi[ok], b[ok], w[ok]
    nv = len(me.vertices)
    contrib = np.bincount(vi, weights=w, minlength=nv)
    order = np.lexsort((-w, vi))
    vi, b, w = vi[order], b[order], w[order]
    rank = np.arange(len(vi)) - np.searchsorted(vi, vi)
    most = int(rank.max()) + 1 if len(rank) else 0
    if most > MAX_BONES:
        return None, f"{int((np.bincount(vi) > MAX_BONES).sum())} vertices weighted to more than {MAX_BONES} bones"
    k = 4 if most <= 4 else 8
    J = np.zeros((nv, k), np.uint16)
    W = np.zeros((nv, k), np.float32)
    J[vi, rank] = b
    W[vi, rank] = w
    live = contrib > 1e-4                                  # less than that and the modifier leaves the vertex alone
    W[live] /= contrib[live, None]
    W[~live] = 0.0
    return (J, W), None


def _group_weights(ob, me, pairs, name):
    """Per Blender vertex its weight in vertex group `name` (0 outside it)."""
    g = ob.vertex_groups.get(name)
    out = np.zeros(len(me.vertices))
    if g is not None:
        vi, gi, w = pairs
        sel = gi == g.index
        out[vi[sel]] = w[sel]
    return out


def _morphs(ob, me, pairs_of):
    """The relative shape keys that can move the mesh: (key block indices, deltas (k, nv, 3) per Blender vertex, Blender
    axes): each key's offset from its own relative key, times its vertex group."""
    sk = me.shape_keys
    nv = len(me.vertices)
    cache = {}

    def co(b):
        if b.name not in cache:
            cache[b.name] = _get(b.data, "co", nv, 3).astype(float)
        return cache[b.name]
    idx, deltas = [], []
    for i, b in enumerate(sk.key_blocks):
        if b == sk.reference_key or b.mute or b.relative_key == b:
            continue
        d = co(b) - co(b.relative_key)
        if b.vertex_group:
            d *= _group_weights(ob, me, pairs_of(), b.vertex_group)[:, None]
        idx.append(i)
        deltas.append(d)
    return idx, (np.array(deltas) if deltas else np.zeros((0, nv, 3)))


def _kind(o):
    """How the bake holds a mesh object: ("static" | "skinned" | "by_frame", its armature, why it is baked frame by
    frame)."""
    sk = o.data.shape_keys
    if sk is not None and not sk.use_relative:
        return "by_frame", None, "absolute shape keys"
    if sk is not None and "mmd_sdef_skinning" in sk.key_blocks and not sk.key_blocks["mmd_sdef_skinning"].mute:
        return "by_frame", None, "mmd_tools SDEF skinning"
    if o.show_only_shape_key and sk is not None:
        return "by_frame", None, "it shows only its active shape key"
    mods = [m for m in o.modifiers if m.show_render and m.type != "PARTICLE_SYSTEM"]
    other = sorted({m.type.lower() for m in mods if m.type != "ARMATURE"})
    if other:
        return "by_frame", None, "modifiers: " + ", ".join(other)
    arms = [m for m in mods if m.type == "ARMATURE" and m.object is not None and m.object.type == "ARMATURE"]
    if not arms:
        return ("by_frame", None, "deformed by its parent armature") if o.parent_type == "ARMATURE" else ("static", None, None)
    m = arms[0]
    why = ("two armature modifiers" if len(arms) > 1 else
           "preserve volume (dual quaternion skinning)" if m.use_deform_preserve_volume else
           "bone envelopes" if m.use_bone_envelopes or not m.use_vertex_groups else
           "a masked armature modifier" if m.vertex_group else
           "a multi-modifier armature" if m.use_multi_modifier else None)
    return ("by_frame", None, why) if why else ("skinned", m.object, None)


def _visible(o):
    return {"camera": o.visible_camera, "shadow": o.visible_shadow, "diffuse": o.visible_diffuse,
            "glossy": o.visible_glossy, "transmission": o.visible_transmission,
            "volume_scatter": o.visible_volume_scatter, "holdout": o.is_holdout, "shadow_catcher": o.is_shadow_catcher}


def _keyed_hide(o):
    ad = o.animation_data
    return bool(ad and ad.action and ad.action.fcurves.find("hide_render") is not None)


def _enabled(view_layer):
    """Names of the objects in collections a render of the view layer includes."""
    out, todo = set(), [view_layer.layer_collection]
    while todo:
        lc = todo.pop()
        if lc.exclude or lc.collection.hide_render:
            continue
        out.update(o.name for o in lc.collection.objects)
        todo.extend(lc.children)
    return out


def _object_rec(o):
    return {"color": list(o.color), "pass_index": o.pass_index}


class _Blob:
    """lit.bin (or a by_frame object's part of it) being written: arrays appended 16-byte aligned, little-endian."""

    def __init__(self, path):
        self.path = path
        self.fh = open(path, "wb")
        self.size = 0

    def _pad(self):
        pad = -self.size % 16
        if pad:
            self.fh.write(b"\0" * pad)
            self.size += pad

    def put(self, arr, dt):
        a = np.ascontiguousarray(arr, DTYPES[dt])
        self._pad()
        at = self.size
        self.fh.write(a.tobytes())
        self.size += a.nbytes
        return at

    def append(self, path):
        """Copy a part in; answers the offset its own offsets are relative to."""
        self._pad()
        at = self.size
        with open(path, "rb") as src:
            shutil.copyfileobj(src, self.fh, 1 << 24)
        self.size += os.path.getsize(path)
        return at

    def close(self):
        self.fh.close()


class _Track:
    """A mesh baked whole (static or skinned): its world matrix, visibility and shape-key values on every frame."""

    def __init__(self, o, kind, arm, n):
        self.ob, self.kind, self.arm = o, kind, arm
        self.M = np.empty((n, 4, 4))
        self.shown = np.zeros(n, bool)
        sk = o.data.shape_keys
        self.kb = np.empty(len(sk.key_blocks), np.float32) if sk is not None else None
        self.keys = np.zeros((n, len(self.kb)), np.float32) if sk is not None else None
        self.truth = {}                                     # check frame -> (world positions, corner normals) glTF
        self.fit = {}                                       # fit frame -> (object-space positions, corner normals)
        self.bends = kind == "skinned" or sk is not None    # its shape changes: Blender bends its normals
        self.geom = self.rec = None
        self.oattr = _ObjectAttrs.of([s.material for s in o.material_slots], n)

    def sample(self, k):
        """Its shape keys' values on frame k (its matrix and visibility are read with every object's: Lit._make_bulk)."""
        self.ob.data.shape_keys.key_blocks.foreach_get("value", self.kb)
        self.keys[k] = self.kb


class _ObjectAttrs:
    """What an object's materials read off it as a whole (Attribute nodes of type OBJECT, INSTANCER, VIEW_LAYER) on every
    frame (module docstring: object_attrs)."""

    def __init__(self, keys, n):
        self.keys = keys
        self.vals = np.zeros((n, len(keys), 4), np.float32)

    @classmethod
    def of(cls, materials, n):
        """For an object with these materials (None: an empty slot); None when they read nothing per object."""
        keys = _object_reads([m.node_tree for m in materials if m is not None and m.use_nodes])
        return cls(keys, n) if keys else None

    def sample(self, k, ob, sc, by=None, psys=None):
        self.vals[k] = [_object_attr(key, ob, sc, by, psys) for key in self.keys]

    def record(self, put):
        out = {}
        for j, key in enumerate(self.keys):
            v = self.vals[:, j]
            still = np.allclose(v, v[:1])
            out[key] = {"offset": put(v[:1] if still else v), "frames": 1 if still else len(v)}
        return out


class _Flex:
    """An object baked frame by frame: its evaluated mesh in its own space, with what its materials read, and its world
    matrix, on each frame it shows; a geometry equal to the last one is not stored again, nor an array equal to the
    last one's, so an object that only moves (type on a car) stores its geometry once."""

    def __init__(self, o, n, part, i, reads, why):
        self.ob, self.reads, self.why_flex = o, reads, why
        self.frames = np.full(n, -1, np.int64)
        self.M = np.tile(np.eye(4), (n, 1, 1))
        self.path, self.blob = os.path.join(part, f".flex{i}.bin"), None      # written from its first geometry on
        self.geoms, self.last, self.mats, self.why = [], None, [], None
        me = o.data if o.type == "MESH" else None
        self.orco = _get(me.vertices, "co", len(me.vertices), 3) if me is not None else None
        data = o.data
        self.texspace = ((np.array(data.texspace_location), np.array(data.texspace_size))
                         if hasattr(data, "texspace_location") else None)
        self.oattr = _ObjectAttrs.of([bpy.data.materials[m] for m in ST.look_materials(o)], n)

    def add(self, k, dg):
        if self.why is not None:
            return
        ev = self.ob.evaluated_get(dg)
        W = np.array(ev.matrix_world)
        if abs(np.linalg.det(W[:3, :3])) < 1e-12:
            return                                               # scaled to nothing: it shows nothing
        try:
            me = ev.to_mesh()
        except RuntimeError as e:
            self.why = f"cannot be evaluated: {e}"
            return
        try:
            if me is None:
                return
            mats = [s.material.name if s.material else None for s in ev.material_slots] or [None]
            orco = self.orco if self.orco is not None and len(self.orco) == len(me.vertices) else None
            g = _geom(me, self.reads, len(mats), orco=orco, texspace=self.texspace, generated="generated" in self.reads)
        finally:
            ev.to_mesh_clear()
        if g is None:
            return
        self.M[k] = W
        for s in g["slots"]:                                     # its slots as indices of the object's merged list
            m = mats[s[0]]
            if m not in self.mats:
                self.mats.append(m)
            s[0] = self.mats.index(m)
        arrays = dict(g["attrs"], **{"=" + s: a for s, a in g["named"].items()}, tris=g["tris"])
        last = self.last
        if (last is not None and last[1]["slots"] == g["slots"] and last[0].keys() == arrays.keys()
                and all(np.array_equal(last[0][key], a) for key, a in arrays.items())):
            self.frames[k] = len(self.geoms) - 1
            return
        at = {}
        for key, a in arrays.items():
            same = last is not None and key in last[0] and np.array_equal(last[0][key], a)
            if not same and self.blob is None:
                self.blob = _Blob(self.path)
            at[key] = last[2][key] if same else self.blob.put(a, "u32" if key == "tris" else "f32")
        rec = {"vertices": int(len(g["src"])), "attributes": {}, "triangles": {"offset": at["tris"], "count": len(g["tris"])},
               "slots": g["slots"]}
        for key, a in g["attrs"].items():
            rec["attributes"][key] = {"offset": at[key], "comp": int(a.shape[1]), "type": "f32"}
        for key, s in g["alias"].items():
            rec["attributes"][key] = {"offset": at["=" + s], "comp": int(g["named"][s].shape[1]), "type": "f32"}
        self.geoms.append(rec)
        self.last = (arrays, rec, at)
        self.frames[k] = len(self.geoms) - 1
        if self.blob is not None and self.blob.size > FLEX_CAP:
            self.why = f"too big to bake on every frame (over {FLEX_CAP // 1000000} MB)"

    def close(self):
        if self.blob is not None:
            self.blob.close()
        self.last = None

    def matrices(self):
        """Its world matrix on every frame, a frame it does not show holding the one before (the first it shows
        before that): what places its geometries, and what the shutter's ends blend."""
        shown = self.frames >= 0
        if not shown.any():
            return self.M[:1]
        idx = np.where(shown, np.arange(len(shown)), 0)
        np.maximum.accumulate(idx, out=idx)
        idx[:int(np.argmax(shown))] = int(np.argmax(shown))
        return self.M[idx]

    def discard(self):
        if self.blob is not None:
            os.remove(self.blob.path)


SCREEN_EXTERNAL = {"GeometryNodeInputSceneTime", "GeometryNodeObjectInfo", "GeometryNodeCollectionInfo",
                   "GeometryNodeInputActiveCamera", "GeometryNodeImageTexture", "GeometryNodeInputImage",
                   "GeometryNodeImageInfo", "GeometryNodeSimulationInput", "GeometryNodeSimulationOutput",
                   "GeometryNodeRepeatInput", "GeometryNodeRepeatOutput"}   # what a word's tree may read off the scene


def _anim(idb):
    """What animates an ID, as JSON: its action's and NLA strips' F-curves (keys, handles, interpolation, easing,
    extrapolation, modifiers), or None; ("drivers", ...) when drivers drive it (they read the scene elsewhere)."""
    ad = getattr(idb, "animation_data", None)
    if ad is None:
        return None
    if len(ad.drivers):
        return "drivers"

    def curves(action):
        if action is None:
            return None
        out = []
        for fc in action.fcurves:
            kp = fc.keyframe_points
            n = len(kp)
            out.append({"path": fc.data_path, "index": fc.array_index, "extrapolation": fc.extrapolation,
                        "mute": fc.mute, "co": _get(kp, "co", n, 2).tolist() if n else [],
                        "left": _get(kp, "handle_left", n, 2).tolist() if n else [],
                        "right": _get(kp, "handle_right", n, 2).tolist() if n else [],
                        "kinds": [[p.interpolation, p.easing, p.handle_left_type, p.handle_right_type, p.back,
                                   p.amplitude, p.period] for p in kp],
                        "modifiers": [_props(m) for m in fc.modifiers]})
        return out
    return {"action": curves(ad.action), "blend": [ad.action_blend_type, ad.action_influence, ad.action_extrapolation],
            "nla": [[t.mute, [[_props(s, depth=0), curves(s.action)] for s in t.strips]] for t in ad.nla_tracks]}


@functools.cache
def _screen_code():
    """The fingerprint of the code that bakes a screen word (mkmmd.cache.code): _ScreenWord, the Lit methods that drive
    and evaluate it and all they reach, fitgeo included; an edit anywhere else in the bake leaves the cache as it is."""
    return CA.code(sys.modules[__name__], "_ScreenWord", "Lit._screen", "Lit._render_settings")


def _screen_key(o, reads, k0, k1, frame0, n):
    """The cache key of a screen word's fitted encoding (`<project>/.mk/cache/screen/<key>.npz`): a SHA-256 of what
    makes its geometry and attributes on each frame of its window (its modifiers and their node trees with every node
    value, the F-curves on them, on the object and on its parents, the object's and its parents' transforms, the fonts'
    bytes, the window, what its materials read, and the code that bakes it: _screen_code), or None when it reads the
    scene elsewhere (drivers, constraints, time, other objects, images, simulation): such a word is baked every time."""
    used = {"groups": set(), "images": set(), "objects": set()}
    mods, trees, fonts = [], {}, []
    for md in o.modifiers:
        mods.append({"type": md.type, "props": _props(md, depth=0), "inputs": {k: _val(md[k]) for k in md.keys()}})
        if md.type == "NODES" and md.node_group is not None:
            used["groups"].add(md.node_group.name)
    todo = sorted(used["groups"])
    while todo:
        name = todo.pop()
        if name in trees:
            continue
        ng = bpy.data.node_groups[name]
        if any(nd.bl_idname in SCREEN_EXTERNAL for nd in ng.nodes):
            return None
        before = set(used["groups"])
        trees[name] = {"graph": graph(ng, used), "anim": _anim(ng)}
        for nd in ng.nodes:
            f = getattr(nd, "font", None)
            if f is not None:
                if f.packed_file is not None:
                    fonts.append([f.name, hashlib.sha256(bytes(f.packed_file.data)).hexdigest()])
                elif f.filepath and f.filepath != "<builtin>":
                    fonts.append([f.name, CA.Path(bpy.path.abspath(f.filepath))])
                else:
                    fonts.append([f.name, "builtin"])
        todo.extend(sorted(used["groups"] - before - set(trees)))
    chain, p = [], o
    while p is not None:
        if len(p.constraints):
            return None
        chain.append({"name": p.name, "type": p.type, "basis": np.array(p.matrix_basis).tolist(),
                      "parent_inverse": np.array(p.matrix_parent_inverse).tolist(), "parent_type": p.parent_type,
                      "anim": _anim(p), "data_anim": _anim(p.data) if p.data is not None else None})
        p = p.parent
    anims = [c["anim"] for c in chain] + [c["data_anim"] for c in chain] + [t["anim"] for t in trees.values()]
    if used["objects"] or used["images"] or "drivers" in anims:
        return None
    data = o.data
    shape = {"vertices": len(data.vertices) if o.type == "MESH" else None,
             "texspace": [list(data.texspace_location), list(data.texspace_size)] if hasattr(data, "texspace_location")
             else None}
    return CA.key("screen", _screen_code(), chain, shape, mods, trees, fonts,
                  [k0, k1, frame0, n], sorted(reads), [NORMAL_KEY, SCREEN_CHECK])


class _ScreenWord:
    """A word of screen type baked on each frame of its window [k0, k1) as fitted geometry (mkmmd.core.fitgeo): its
    evaluated mesh in the world, corners merged by normal (NORMAL_KEY), with what its materials read. A frame of the
    last drawn one's topology (the same corners, faces, materials and corner-data split) is read straight off the
    evaluated mesh; any other is drawn whole (`_geom`). Its check frames (first, middle, last) are drawn whole as well
    and kept, sampled, to measure the bake against (`check`). With a `store` (mkmmd.cache.Cache) and a `key` it takes
    the encoding the store holds for that key, and is never evaluated; else it bakes and leaves its encoding there."""

    def __init__(self, o, n, part, i, reads, k0, k1, store=None, key=None):
        self.ob, self.reads, self.k0, self.k1 = o, reads, k0, k1
        self.path, self.blob = os.path.join(part, f".screen{i}.bin"), None
        self.word = FG.Word(n, self._put)
        self.mats, self.why, self.topo, self.rec, self.bytes = [], None, None, None, 0
        self.check_k = sorted({k0, (k0 + k1) // 2, k1 - 1})
        self.truth = {}                       # check frame -> (materialised vertices, positions, normals, attrs, solid)
        self.t_eval = self.t_fit = 0.0
        me = o.data if o.type == "MESH" else None
        self.orco = _get(me.vertices, "co", len(me.vertices), 3) if me is not None else None
        data = o.data
        self.texspace = ((np.array(data.texspace_location), np.array(data.texspace_size))
                         if hasattr(data, "texspace_location") else None)
        self.quick = not any(r == "generated" or r.startswith("tangent:") for r in reads)   # what a quick read lacks
        self.store, self.key = store, key
        self.cached = store is not None and key is not None and self._load()

    def _load(self):
        """Take the store's encoding of this word: its part of lit.bin, its record, materials, stats and checks."""
        z = self.store.load_npz("screen", self.key)
        if z is None:
            return False
        meta = json.loads(z["meta"].tobytes().decode("utf-8"))
        z["part"].tofile(self.path)
        w = self.word
        w.geoms, w.stats = meta["geoms"], meta["stats"]
        w.frames[:], w.poses[:] = meta["frames"], meta["poses"]
        self.mats = meta["mats"]
        for j, (k, keys) in enumerate(meta["truth"]):
            self.truth[k] = (z[f"t{j}_idx"], z[f"t{j}_pos"], z[f"t{j}_nrm"],
                             {key: z[f"t{j}_a{m}"] for m, key in enumerate(keys)}, z[f"t{j}_solid"])
        return True

    def _save(self):
        """Leave this word's encoding in the store under its key."""
        w = self.word
        meta = {"geoms": w.geoms, "frames": w.frames.tolist(), "poses": w.poses.tolist(), "stats": w.stats,
                "mats": self.mats, "truth": [[int(k), list(t[3])] for k, t in sorted(self.truth.items())]}
        arrays = {"part": np.fromfile(self.path, np.uint8),
                  "meta": np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)}
        for j, (k, (idx, pos, nrm, attrs, solid)) in enumerate(sorted(self.truth.items())):
            arrays.update({f"t{j}_idx": idx, f"t{j}_pos": pos, f"t{j}_nrm": nrm, f"t{j}_solid": solid})
            arrays.update({f"t{j}_a{m}": a for m, a in enumerate(attrs.values())})
        self.store.save_npz("screen", self.key, arrays)

    def _put(self, a, dt):
        if self.blob is None:
            self.blob = _Blob(self.path)
        return self.blob.put(a, dt)

    def add(self, k, dg):
        if self.why is not None or self.cached:
            return
        t0 = time.time()
        ev = self.ob.evaluated_get(dg)
        try:
            me = ev.to_mesh()
        except RuntimeError as e:
            self.why = f"cannot be evaluated: {e}"
            return
        try:
            if me is not None:
                self._frame(k, ev, me)
        finally:
            ev.to_mesh_clear()
        self.t_eval += time.time() - t0
        if self.blob is not None and self.blob.size > FLEX_CAP:
            self.why = f"too big to bake on every frame (over {FLEX_CAP // 1000000} MB)"

    def _frame(self, k, ev, me):
        mats = [s.material.name if s.material else None for s in ev.material_slots] or [None]
        W = np.array(ev.matrix_world)
        quick = self._quick(me, W, mats) if self.quick and self.topo is not None else None
        g = None
        if quick is None or k in self.check_k:
            orco = self.orco if self.orco is not None and len(self.orco) == len(me.vertices) else None
            g = _geom(me, self.reads, len(mats), world=W, orco=orco, texspace=self.texspace,
                      generated="generated" in self.reads, merge=True)
            if g is None:
                return                                          # nothing to draw: hidden
            for s in g["slots"]:                                # its slots as indices of the object's merged list
                if mats[s[0]] not in self.mats:
                    self.mats.append(mats[s[0]])
                s[0] = self.mats.index(mats[s[0]])
        t0 = time.time()
        if quick is None:
            attrs = {key: g["named"][s] for key, s in g["alias"].items()}
            if "generated" in g["attrs"]:
                attrs["generated"] = g["attrs"]["generated"]
            self.word.add(k, g["attrs"]["position"], g["attrs"]["normal"], attrs, topo=(g["tris"], g["slots"], g["src"]))
            self.topo = self._topology(me, g, mats)
        else:
            self.word.add(k, *quick)
        self.t_fit += time.time() - t0
        if k in self.check_k:
            self._keep(k, g)

    def _topology(self, me, g, mats):
        """What a later frame must share with this one to be read quickly."""
        nl, npoly = len(me.loops), len(me.polygons)
        tot = _loop_totals(me)
        return {"n": (len(me.vertices), nl, npoly), "mats": mats, "dom": me.normals_domain, "cv": _corner_verts(me),
                "tot": tot, "cpoly": np.repeat(np.arange(npoly, dtype=np.int32), tot), "mi": _face_materials(me),
                "tl": _get(me.loop_triangles, "loops", len(me.loop_triangles), 3, np.int32) if (tot > 3).any() else None,
                "first": g["first"], "corner": g["corner"], "src": g["src"], "alias": g["alias"]}

    def _quick(self, me, W, mats):
        """(positions, normals, attributes) of a frame of the last drawn one's topology, read off the evaluated mesh as
        `_geom` would draw them, or None (anything differs: draw it whole)."""
        t = self.topo
        nv, nl, npoly = len(me.vertices), len(me.loops), len(me.polygons)
        if mats != t["mats"] or (nv, nl, npoly) != t["n"] or me.normals_domain != t["dom"]:
            return None
        if abs(np.linalg.det(W[:3, :3])) < 1e-12:
            return None
        cv = _corner_verts(me)
        if not (np.array_equal(cv, t["cv"]) and np.array_equal(_loop_totals(me), t["tot"])
                and np.array_equal(_face_materials(me), t["mi"])):
            return None
        if t["tl"] is not None:                                 # n-gons: their triangles follow their shape
            me.calc_loop_triangles()
            if not np.array_equal(_get(me.loop_triangles, "loops", len(me.loop_triangles), 3, np.int32), t["tl"]):
                return None
        first, corner = t["first"], t["corner"]
        cn = _corner_normals(me, cv, t["cpoly"])
        if t["dom"] != "POINT":                                 # the corners of each vertex still round alike
            key = _normal_key(cn)
            if not np.array_equal(key, key[first][corner]):
                return None
        data, alias = _corner_data(me, self.reads, cv, t["cpoly"])
        if alias != t["alias"] or any(not np.array_equal(a, a[first][corner]) for a in data.values()):
            return None
        co = _get(me.attributes["position"].data, "vector", nv, 3).astype(float)
        pos = ((co @ W[:3, :3].T + W[:3, 3]) @ AXES3.T)[t["src"]]
        nor = cn[first].astype(float) @ np.linalg.inv(W[:3, :3])
        nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)
        return (pos.astype(np.float32), (nor @ AXES3.T).astype(np.float32),
                {key: data[s][first] for key, s in alias.items()})

    def _keep(self, k, g):
        """A check frame drawn whole: a stride of the vertices of the geometry that holds it, each read off this
        drawing at the corner it was made of (a later drawing may join corners the first one kept apart), and which of
        them a triangle that is not a sliver uses (a sliver's normal is float32 noise)."""
        seg = self.word.seg
        L = seg.layout
        idx = np.arange(0, len(L.src), max(1, len(L.src) // SCREEN_CHECK))
        at = g["corner"][self.topo["first"][L.src[idx]]]
        attrs = {key: g["named"][s][at] for key, s in g["alias"].items()}
        self.truth[k] = (idx, g["attrs"]["position"][at], g["attrs"]["normal"][at], attrs, seg.solid[idx])

    def done(self):
        """Its window is over (or it is left out): drop what fitting its frames needed, keeping its records and the
        check frames' samples (a clip's hundreds of words would otherwise hold gigabytes to the end of the loop)."""
        self.word.done()
        self.topo = None

    def close(self):
        """The loop is over: close its part of lit.bin and, when it baked one whole, leave the encoding in the store."""
        if self.blob is not None:
            self.blob.close()
            if self.store is not None and self.key is not None and self.why is None and self.word.geoms:
                self._save()
        self.topo = None

    def discard(self):
        if os.path.exists(self.path):
            os.remove(self.path)


class _Inst:
    """An instance (geometry nodes, a collection instance, particles) of a mesh object: its world matrix per frame, and
    what its materials read per object."""

    def __init__(self, by, of, pid, n):
        self.by, self.of, self.pid = by, of, pid
        self.M = np.full((n, 4, 4), np.nan)
        self.seen = np.zeros(n, bool)
        self.oattr = _ObjectAttrs.of([s.material for s in bpy.data.objects[of].material_slots], n)


# ================================================================================================= the section
class Lit:
    """The lit section (module docstring): made before the bake's frame loop, `sample`d on every frame, `capture`d on
    the export frame before the glTF export rewires the MMD materials, written by `finish`, checked by `check`."""

    def __init__(self, sc, n, cams, part, log, ctx=None, outputs=()):
        t0 = time.time()
        self.sc, self.n, self.part, self.log = sc, n, part, log
        self.frame0 = ctx.frame0 if ctx is not None else sc.frame_start
        self.view = view(sc, part)
        self.t_lut = time.time() - t0
        self.check_k = sorted({0, n // 2, n - 1})
        self.fit_k = sorted({int(round(x)) for x in np.linspace(0, n - 1, 10)[1:-1]} - set(self.check_k))
        enabled = _enabled(bpy.context.view_layer)
        self.lights, self.tracks, self.flex, self.left, self.instancers = [], [], [], {}, set()
        self.watch = {}                                   # name -> (object, shows every frame whatever hide_render)
        flex_n = 0
        for o in sc.objects:
            aspect = "mk_aspect" in o.keys()
            if o.name not in enabled or "mk_screen" in o.keys() or not (ST.renders(o) or aspect):
                continue
            self.watch[o.name] = (o, aspect and not _keyed_hide(o))
            if o.type == "LIGHT":
                self.lights.append((o, np.zeros((n, LIGHT_FLOATS), np.float32)))
                continue
            if o.type in UNDRAWN:
                self.left[o.name] = (UNDRAWN[o.type], o.name)
                continue
            if self._instancer(o):
                self.instancers.add(o.name)
            if o.type == "MESH":
                for ps in o.particle_systems:
                    if ps.settings.render_type == "PATH":
                        self.left[f"{o.name} ({ps.name})"] = ("hair particles", o.name)
                if o.particle_systems and not any(ps.settings.use_render_emitter for ps in o.particle_systems):
                    continue
                kind, arm, why = _kind(o)
            elif o.type in EVALUATED:
                kind, arm, why = "by_frame", None, f"a {o.type.lower()} object"
            else:
                continue
            if kind == "by_frame":
                reads = _reads([bpy.data.materials[m].node_tree for m in ST.look_materials(o)
                                if bpy.data.materials[m].use_nodes])
                self.flex.append(_Flex(o, n, part, flex_n, reads, why))
                flex_n += 1
            else:
                self.tracks.append(_Track(o, kind, arm, n))
        self.shown_arr = np.zeros((len(self.watch), n), bool)         # filled for all of them at once (`_bulk`)
        self.shown = {name: self.shown_arr[i] for i, name in enumerate(self.watch)}
        self.track_M, self.track_shown = np.empty((len(self.tracks), n, 4, 4)), np.zeros((len(self.tracks), n), bool)
        for i, t_ in enumerate(self.tracks):
            t_.M, t_.shown = self.track_M[i], self.track_shown[i]
        self.bulk = None
        self.saved, self.render_changes = [], {}          # what the bake changes for its loop, put back by `restore`
        for fx in self.flex:                              # the lit frames are renders: evaluated as a render does
            self._render_settings(fx.ob)
        self.screen = []                                  # screen type: a _ScreenWord per word
        store = CA.Cache(ctx.cache) if ctx is not None and getattr(ctx, "cache", None) else None
        t0_key = time.time()
        for o in sc.objects:
            if "mk_screen" not in o.keys() or (o.type != "MESH" and o.type not in EVALUATED):
                continue
            show = o.get("mk_show")
            k0, k1 = (max(int(show[0]) - self.frame0, 0), min(int(show[1]) - self.frame0, n)) if show else (0, n)
            if k0 < k1:
                reads = _reads([bpy.data.materials[m].node_tree for m in ST.look_materials(o)
                                if bpy.data.materials[m].use_nodes])
                key = _screen_key(o, reads, k0, k1, self.frame0, n) if store is not None else None
                sw = _ScreenWord(o, n, part, len(self.screen), reads, k0, k1, store, key)
                self.screen.append(sw)
                self._render_settings(o)
                self.saved.append((o, "hide_viewport", o.hide_viewport))   # evaluated inside its window only
                o.hide_viewport = k0 > 0 or sw.cached     # one that shows from the first frame: evaluated from it
        self.t_screen_keys = time.time() - t0_key
        self.holder = self._materials_holder()
        self.screen_doc = self._screen_doc(sc, outputs) if self.screen else None
        self.points = {}                                  # the layers' points: key -> expression, shots, world xyz
        self.points_error = None
        if ctx is not None:
            try:
                pl = TR.plan(ctx.data, ctx.fps, ctx.frame0, ctx.palette)
                dem = TR.demands(pl) if TR.has_effects(pl) else {}
            except TR.TransitionError as e:
                dem, self.points_error = {}, str(e)
            for f, items in dem.items():
                for it in items:
                    if it["kind"] == "point" and 0 <= int(f) - self.frame0 < n:
                        p = self.points.setdefault(it["key"], {"expr": it["expr"], "shots": {}, "errors": {},
                                                               "code": compile(it["expr"], "<mk point>", "eval"),
                                                               "xyz": np.full((n, 3), np.nan, np.float32)})
                        p["shots"][int(f) - self.frame0] = it["shot"]
        self.cams = list(cams)
        self.cam_rows = {c: np.zeros((n, CAMERA_FLOATS), np.float32) for c in cams}
        self.chans = []                                   # (owner kind, owner name, node, socket, io, values)
        for kind, coll in (("material", bpy.data.materials), ("world", bpy.data.worlds), ("group", bpy.data.node_groups)):
            for owner in coll:
                nt = owner if kind == "group" else owner.node_tree
                if nt is None or (kind == "group" and nt.bl_idname != "ShaderNodeTree"):
                    continue
                for node, s, io in _channels(nt, None if kind == "group" else owner):
                    size = len(s.default_value) if hasattr(s.default_value, "__len__") else 1
                    self.chans.append((kind, owner.name, node, s, io, np.zeros((n, size), np.float32)))
        self.node_objects = {}                           # objects node trees point at: their world matrix per frame
        trees = [m.node_tree for m in bpy.data.materials if m.node_tree] + [w.node_tree for w in bpy.data.worlds
                                                                               if w.node_tree]
        trees += [g for g in bpy.data.node_groups if g.bl_idname == "ShaderNodeTree"]
        for nt in trees:
            for node in nt.nodes:
                ob = getattr(node, "object", None)
                if isinstance(ob, bpy.types.Object):
                    self.node_objects[ob.name] = np.empty((n, 4, 4))
        self.inst = {}
        self.t_sample = 0.0
        self.t_parts = dict.fromkeys(("records", "tracks", "instances", "by_frame", "screen", "points", "truth"), 0.0)
        self.docs = None
        log(f"bake: lit: {len(self.tracks)} meshes, {len(self.flex)} by frame, {len(self.lights)} lights, "
            f"{len(self.chans)} animated node values, view LUT in {self.t_lut:.1f}s")

    def _render_settings(self, o):
        """Evaluate `o` as a render does for the bake's loop: modifiers shown as in renders, Subdivision Surface and
        Multires at their render levels, curves and metaballs at their render resolution (put back by `restore`)."""
        changes = []

        def put(owner, attr, value, what):
            if getattr(owner, attr) != value:
                self.saved.append((owner, attr, getattr(owner, attr)))
                setattr(owner, attr, value)
                changes.append(what)
        for md in o.modifiers:
            put(md, "show_viewport", md.show_render, f"{md.name}: shown as in renders")
            if md.type in ("SUBSURF", "MULTIRES"):
                put(md, "levels", md.render_levels, f"{md.name}: level {md.render_levels}")
        d = o.data
        for attr in ("resolution_u", "resolution_v"):
            if o.type in ("CURVE", "FONT", "SURFACE") and getattr(d, "render_" + attr, 0):
                put(d, attr, getattr(d, "render_" + attr), f"{attr} {getattr(d, 'render_' + attr)}")
        if o.type == "META" and d.render_resolution > 0:
            put(d, "resolution", d.render_resolution, f"resolution {d.render_resolution}")
        if changes:
            self.render_changes[o.name] = changes

    def _materials_holder(self):
        """An empty mesh that wears every screen word's materials, shown for the bake's loop: a word is hidden from the
        depsgraph outside its window and the whole clip when its encoding comes from the cache, and a material no shown
        object wears is not evaluated, so its drivers (a breathing palette's colours) would stop on whatever value they
        last had. None without screen type."""
        mats = []
        for sw in self.screen:
            for name in ST.look_materials(sw.ob):
                if name in bpy.data.materials and name not in mats:
                    mats.append(name)
        if not mats:
            return None
        me = bpy.data.meshes.new("mk_bake_materials")
        for name in mats:
            me.materials.append(bpy.data.materials[name])
        ob = bpy.data.objects.new("mk_bake_materials", me)
        ob.hide_render = True
        self.sc.collection.objects.link(ob)
        return ob

    def restore(self):
        """Put back what the bake changed for its loop (render settings, screen type shown to the depsgraph) and remove
        its materials holder."""
        for owner, attr, value in reversed(self.saved):
            setattr(owner, attr, value)
        self.saved = []
        if self.holder is not None:
            me = self.holder.data
            bpy.data.objects.remove(self.holder)
            bpy.data.meshes.remove(me)
            self.holder = None

    def _screen_doc(self, sc, outputs):
        """The screen layer's pass (styles.Looks._screen_pass): its view transform as a LUT (Standard, no look), the
        camera per output, the render settings and the file format."""
        vs = sc.view_settings
        keys = ("look", "view_transform", "exposure", "gamma", "use_curve_mapping")
        saved = [(k, getattr(vs, k)) for k in keys]
        try:
            for k, v in zip(keys, ("None", "Standard", 0.0, 1.0, False)):
                setattr(vs, k, v)
            lut, zero, floor = _capture(sc, self.part)
        finally:
            for k, v in reversed(saved):                   # the look last: setting the transform resets it
                setattr(vs, k, v)
        lut.astype("<f4").tofile(os.path.join(self.part, "screen_lut.bin"))
        M = np.eye(4)
        M[2, 3] = 10.0                                     # at z = 10 looking down -z, +y up (Blender's axes)
        samples = {}
        for e in json.loads(sc.get("mk_shots", "[]")):
            for out, look in (e.get("styles") or {}).items():
                for spec in (look or {}).values():
                    if isinstance(spec, dict) and spec.get("samples"):
                        samples.setdefault(e["name"], {})[out] = int(spec["samples"])
        for w in json.loads(sc.get("mk_medium") or "[]"):
            for out, look in (w.get("styles") or {}).items():
                for spec in (look or {}).values():
                    if isinstance(spec, dict) and spec.get("samples"):
                        samples.setdefault(f"medium {w['index']}", {})[out] = int(spec["samples"])
        return {"lut": {"file": "screen_lut.bin", "n": LUT_N, "lo": LUT_LO, "hi": LUT_HI}, "zero": zero.tolist(),
                "floor": floor.tolist(), "view": {"transform": "Standard", "look": "None", "exposure": 0.0, "gamma": 1.0,
                                                  "curve": False, "display": sc.display_settings.display_device},
                "cameras": {o["name"]: {"type": "ORTHO", "ortho_scale": SR.ortho_scale(tuple(o["size"])),
                                        "sensor_fit": "AUTO", "clip": [0.1, 100.0], "size": list(o["size"]),
                                        "matrix": _affine(_gl4(M)).tolist()} for o in outputs},
                "pass": {"engine": "BLENDER_EEVEE_NEXT", "film_transparent": True, "lights": False, "world": "black",
                         "motion_blur": False, "dither": 0.0, "compositing": False, "use_raytracing": False,
                         "samples": {"default": 16, "looks": samples}},
                "format": {"file_format": "PNG", "color_mode": "RGBA", "color_depth": 8, "alpha": "STRAIGHT"},
                "meshes": []}

    def _screen(self, k):
        """Screen type on frame k: each word is shown to the depsgraph from the frame before its window to its end
        (hundreds of words would be evaluated on every frame otherwise; one shown only now would miss this frame's
        animation until the next frame change), baked frame by frame inside its window as fitted geometry. The layer
        as a whole stops at SCREEN_CAP: what is not baked by then is Blender's to draw."""
        dg = bpy.context.evaluated_depsgraph_get()
        if sum(sw.blob.size for sw in self.screen if sw.blob is not None) > SCREEN_CAP:
            for sw in self.screen:                         # what is not baked yet is Blender's to draw
                if sw.why is None and k < sw.k1:
                    sw.why = f"the screen layer is over the bake's cap ({SCREEN_CAP // 1000000} MB)"
        for sw in self.screen:
            if sw.k0 <= k < sw.k1:
                sw.add(k, dg)
                if k == sw.k1 - 1 or sw.why is not None:
                    sw.done()                              # its window is over: what fitted it goes (hundreds of words)
            want = sw.why is not None or sw.cached or not (sw.k0 <= k + 1 < sw.k1)    # for the next frame's evaluation
            if sw.ob.hide_viewport != want:
                sw.ob.hide_viewport = want

    @staticmethod
    def _instancer(o):
        if o.instance_type == "COLLECTION" and o.instance_collection is not None:
            return True
        if o.instance_type in ("VERTS", "FACES") and o.children:
            return True
        if any(m.type == "NODES" and m.show_render for m in o.modifiers):
            return True
        return any(ps.settings.render_type in ("OBJECT", "COLLECTION") for ps in getattr(o, "particle_systems", ()))

    # ------------------------------------------------------------------------------------------------- every frame
    def sample(self, k, dg):
        t0 = time.time()
        t = time.perf_counter()
        b = self.bulk or self._make_bulk()
        bpy.data.objects.foreach_get("matrix_world", b["m"])
        bpy.data.objects.foreach_get("hide_render", b["h"])
        W = b["m"].reshape(-1, 4, 4)
        self.shown_arr[:, k] = b["always"] | ~b["h"][b["watch"]]
        for o, rows in self.lights:
            rows[k] = self._light_row(o, self.shown[o.name][k])
        for c in self.cams:
            self.cam_rows[c][k] = self._camera_row(bpy.data.objects[c])
        for *_, s, io, vals in self.chans:
            v = s.default_value
            vals[k] = v[:] if hasattr(v, "__len__") else v
        for name, M in self.node_objects.items():
            M[k] = W[b["at"][name]].T
        t = self._tick("records", t)
        self.track_M[:, k] = W[b["tracks"]].transpose(0, 2, 1)
        self.track_shown[:, k] = self.shown_arr[b["track_watch"], k]
        for t_ in b["keyed"]:
            t_.sample(k)
        for h in b["attrs"]:
            h.oattr.sample(k, h.ob, self.sc)
        t = self._tick("tracks", t)
        if self.instancers:
            self._instances(k, dg)
        t = self._tick("instances", t)
        for fx in self.flex:
            if self.shown[fx.ob.name][k]:
                fx.add(k, dg)
        t = self._tick("by_frame", t)
        if self.screen:
            self._screen(k)
        t = self._tick("screen", t)
        due = [p for p in self.points.values() if k in p["shots"]]
        if due:
            ns = _namespace(None)                         # as mk render's point layer evaluates them
            ns["frame"] = self.frame0 + k
            for p in due:
                try:
                    p["xyz"][k] = AXES3 @ np.array(tuple(eval(p["code"], ns)), float)   # noqa: S307 - the project's own
                except Exception as e:  # noqa: BLE001 - a failing expression is the project's: recorded with its frames
                    p["errors"].setdefault(str(e), []).append(self.frame0 + k)
        t = self._tick("points", t)
        if k in self.check_k or k in self.fit_k:
            self._truth(k, dg)
        self._tick("truth", t)
        self.t_sample += time.time() - t0

    def _make_bulk(self):
        """Where each object the frames sample sits among bpy.data.objects, so that one foreach_get of every object's
        matrix and visibility a frame serves them all (reading them one by one took 40 ms a frame with a thousand
        props); made on the first frame, when the objects the loop sees are all there."""
        order = {o.name: i for i, o in enumerate(bpy.data.objects)}
        pos = {name: i for i, name in enumerate(self.watch)}
        self.bulk = {"m": np.empty(16 * len(order), np.float32), "h": np.empty(len(order), bool),
                     "watch": np.array([order[name] for name in self.watch], int),
                     "always": np.array([always for _, always in self.watch.values()], bool),
                     "at": {name: order[name] for name in self.node_objects},
                     "tracks": np.array([order[t_.ob.name] for t_ in self.tracks], int),
                     "track_watch": np.array([pos[t_.ob.name] for t_ in self.tracks], int),
                     "keyed": [t_ for t_ in self.tracks if t_.kb is not None],
                     "attrs": [h for h in self.tracks + self.flex if h.oattr is not None]}
        return self.bulk

    def _tick(self, part, t):
        """Add the time since `t` to `part` of the frames' sampling; now."""
        now = time.perf_counter()
        self.t_parts[part] += now - t
        return now

    @staticmethod
    def _light_row(o, shown):
        L = o.data
        M = np.array(o.matrix_world)
        X, Y, Z = (AXES3 @ M[:3, i] for i in range(3))      # its own (Blender) axes, in the world, glTF's axes
        sx = sy = 0.0
        if L.type == "AREA":
            sx = L.size * float(np.linalg.norm(X))
            sy = (L.size_y if L.shape in ("RECTANGLE", "ELLIPSE") else L.size) * float(np.linalg.norm(Y))
        spot = (L.spot_size, L.spot_blend) if L.type == "SPOT" else (0.0, 0.0)
        radius = L.angle if L.type == "SUN" else L.shadow_soft_size
        return [*(AXES3 @ M[:3, 3]), *_unit(-Z), *_unit(X), *_unit(Y), *L.color, L.energy, sx, sy, *spot, radius,
                float(shown)]

    @staticmethod
    def _camera_row(o):
        cd = o.data
        dof = cd.dof
        dist = dof.focus_distance
        tgt = dof.focus_object
        if tgt is not None:
            W = tgt.matrix_world
            pb = tgt.pose.bones.get(dof.focus_subtarget) if tgt.type == "ARMATURE" and dof.focus_subtarget else None
            p = (W @ pb.matrix).translation if pb is not None else W.translation
            z = o.matrix_world.col[2].xyz.normalized()
            dist = abs(z.dot(o.matrix_world.translation - p))
        return [float(dof.use_dof), dist, dof.aperture_fstop, dof.aperture_blades, dof.aperture_rotation,
                dof.aperture_ratio, cd.ortho_scale, cd.sensor_width]

    def _instances(self, k, dg):
        for di in dg.object_instances:
            if not di.is_instance:
                continue
            par, src = di.parent, di.instance_object
            if par is None or src is None or par.name not in self.instancers or src.name == par.name:
                continue                                  # its own geometry (a curve's mesh): baked with the object
            key = (par.name, src.name, tuple(di.persistent_id))
            rec = self.inst.get(key)
            if rec is None:
                rec = self.inst[key] = _Inst(par.name, src.name, [i for i in di.persistent_id if i != 2147483647], self.n)
                if (rec.oattr is not None and any(a.startswith("INSTANCER:") for a in rec.oattr.keys)
                        and any(m.type == "NODES" and m.show_render for m in par.original.modifiers)):
                    self.left[par.name] = ("Attribute nodes of type Instancer on geometry-nodes instances (their "
                                           "instance attributes)", par.name)
            rec.M[k] = np.array(di.matrix_world)
            rec.seen[k] = True
            if rec.oattr is not None:
                rec.oattr.sample(k, src.original, self.sc, par.original, di.particle_system)

    def _truth(self, k, dg):
        """Blender's evaluated mesh of every mesh baked whole: on a check frame its world positions per Blender vertex
        and world corner normals (glTF's axes), on a fit frame a deforming one's object-space positions and corner
        normals (Blender's)."""
        for t in self.tracks:
            if k not in self.check_k and not t.bends:
                continue
            ev = t.ob.evaluated_get(dg)
            me = ev.to_mesh()
            try:
                co = _get(me.vertices, "co", len(me.vertices), 3).astype(float)
                cn = _get(me.corner_normals, "vector", len(me.loops), 3).astype(float)
            finally:
                ev.to_mesh_clear()
            if k in self.fit_k:
                t.fit[k] = (co, cn)
            M = np.array(t.ob.matrix_world)
            if k not in self.check_k or abs(np.linalg.det(M[:3, :3])) < 1e-12:
                continue
            n = cn @ np.linalg.inv(M[:3, :3])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            t.truth[k] = (((co @ M[:3, :3].T + M[:3, 3]) @ AXES3.T), n @ AXES3.T)

    # ------------------------------------------------------------------------------------------------- export frame
    def capture(self):
        """Read the node graphs, images and meshes (on the export frame, before the glTF export rewires materials)."""
        t0 = time.time()
        for fx in self.flex + self.screen:
            fx.close()
        mats = []

        def add(names):
            for m in names:
                if m is not None and m not in mats:
                    mats.append(m)
        for t in self.tracks:
            if t.shown.any():
                add(s.material.name if s.material else None for s in t.ob.material_slots)
        sources = sorted({r.of for r in self.inst.values()})
        for name in sources:
            add(s.material.name if s.material else None for s in bpy.data.objects[name].material_slots)
        for fx in self.flex + self.screen:
            add(fx.mats)
        used = {"groups": set(), "images": set(), "objects": set()}
        self.mat_docs = {}
        for m in mats:
            mat = bpy.data.materials[m]
            self.mat_docs[m] = {"settings": _props(mat, skip=_skip("id") | {"node_tree", "grease_pencil", "preview"}),
                                "tree": graph(mat.node_tree, used) if mat.use_nodes and mat.node_tree else None,
                                "anim": []}
        w = self.sc.world
        self.world_doc = None
        if w is not None:
            self.world_doc = {"name": w.name, "use_nodes": w.use_nodes, "color": list(w.color),
                              "props": _props(w, skip=_skip("id") | {"node_tree", "preview", "cycles", "cycles_visibility",
                                                                     "light_settings", "mist_settings"}),
                              "visibility": {k: getattr(w.cycles_visibility, k)
                                             for k in ("camera", "diffuse", "glossy", "transmission", "shadow", "scatter")},
                              "tree": graph(w.node_tree, used) if w.use_nodes and w.node_tree else None, "anim": []}
        self.group_docs, todo = {}, sorted(used["groups"])
        while todo:
            name = todo.pop()
            if name in self.group_docs:
                continue
            ng = bpy.data.node_groups[name]
            before = set(used["groups"])
            self.group_docs[name] = {"interface": _interface(ng), "tree": graph(ng, used), "anim": []}
            todo.extend(sorted(used["groups"] - before - set(self.group_docs)))
        self.image_docs = {name: _image(bpy.data.images[name], self.part, i)
                           for i, name in enumerate(sorted(used["images"]))}
        self.node_objects = {k: v for k, v in self.node_objects.items() if k in used["objects"]}
        for t in self.tracks:                              # the meshes baked whole, as they stand in their rest
            if t.shown.any():
                self._capture_track(t)
        self.sources = {}
        for name in sources:
            self.sources[name] = self._capture_source(bpy.data.objects[name])
        self.t_capture = time.time() - t0

    def _capture_track(self, t):
        o = t.ob
        me = o.data
        reads = _reads([s.material.node_tree for s in o.material_slots if s.material and s.material.use_nodes])
        sk = me.shape_keys
        nv, nl = len(me.vertices), len(me.loops)
        base = _get(sk.reference_key.data, "co", nv, 3) if sk is not None else None
        fans = None
        if t.bends and nl:                                  # it deforms: its normals bend as Blender bends them
            cv = _get(me.loops, "vertex_index", nl, 1, np.int32)
            tot = _get(me.polygons, "loop_total", len(me.polygons), 1, np.int32)
            cpoly = np.repeat(np.arange(len(tot), dtype=np.int32), tot)
            fans = _fans(me, cv, cpoly)
        g = _geom(me, reads, len(o.material_slots), base=base, fans=None if fans is None else fans[0])
        if g is None:
            t.geom = None
            return
        if fans is not None:
            fan, nfan, smooth_e, cnext, cprev, ce = fans
            nface, ns = len(me.polygons), len(g["src"])
            b = BD.fit(_get(me.vertices, "co", nv, 3).astype(float), _get(me.corner_normals, "vector", nl, 3), cv, ce,
                       cpoly, nface, fan, nfan, smooth_e, cnext, cprev, [t.fit[k] for k in sorted(t.fit)])
            anyv = np.zeros(nv, np.int64)
            anyv[g["src"]] = np.arange(ns)                 # a vertex of each of Blender's, for its position
            per = np.zeros((ns, 3))
            per[g["corner"]] = np.stack([fan, b["alpha"], b["beta"]], 1)     # one fan and one normal per vertex
            tri_face = cpoly[g["tri_loops"][:, 0]]

            def csr(keys, n):
                return np.r_[0, np.cumsum(np.bincount(keys, minlength=n))]
            g["bend"] = {"fan": per[:, 0].astype(np.int64), "alpha_beta": per[:, 1:], "center": anyv[b["vertex"]],
                         "ref": anyv[b["ref"]], "other": anyv[b["other"]],
                         "corners": (csr(fan, nfan), np.stack([cpoly, anyv[cv[cprev]], anyv[cv[cnext]]],
                                                              1)[np.argsort(fan, kind="stable")].ravel()),
                         "edges": (csr(b["edge_fan"], nfan), anyv[b["edge_far"]]),
                         "faces": (csr(tri_face, nface), np.argsort(tri_face, kind="stable"))}
        pairs = []

        def pairs_of():
            if not pairs:
                pairs.append(_pairs(me))
            return pairs[0]
        if t.kind == "skinned":
            skin, why = _skin(o, me, t.arm, pairs_of())
            if skin is None:
                self.left[o.name] = (why, o.name)
                t.geom = None
                return
            g["skin"] = (skin[0][g["src"]], skin[1][g["src"]])
        if sk is not None:
            idx, d = _morphs(o, me, pairs_of)
            live = [j for j, i in enumerate(idx) if np.any(t.keys[:, i] != 0.0)]     # keys that move it in the clip
            g["morphs"] = ([sk.key_blocks[idx[j]].name for j in live], [idx[j] for j in live],
                           (d[live][:, g["src"]] @ AXES3.T).astype(np.float32))
        t.geom = g

    def _capture_source(self, o):
        """An instanced object's mesh in its own space, or the reason it cannot be held."""
        if o.type != "MESH":
            return f"instances of a {o.type.lower()} object"
        kind, _, why = _kind(o)
        if kind != "static" or o.data.shape_keys is not None:
            return f"instances of {o.name}, which deforms ({why or kind})"
        reads = _reads([s.material.node_tree for s in o.material_slots if s.material and s.material.use_nodes])
        return _geom(o.data, reads, len(o.material_slots)) or "instances of an empty mesh"

    # ------------------------------------------------------------------------------------------------- writing
    def finish(self, put, armatures, looks=()):
        """Write lit.bin and the per-frame values (`put` adds float32 to bake.bin, answering the FLOAT offset); answers
        the "lit" record. `armatures`: bake.json's armature entries; `looks`: the names bake.json's `objects` records."""
        t0 = time.time()
        n = self.n
        blob = _Blob(os.path.join(self.part, "lit.bin"))
        arm_names = {a["name"] for a in armatures}

        def matrix(M):
            m = _affine(_gl4(M))
            still = np.allclose(m, m[:1], atol=1e-7)
            return {"offset": put(m[:1] if still else m), "frames": 1 if still else n}

        def shown(flags):
            return None if flags.all() else put(flags.astype(np.float32))

        def oattrs(h):
            return h.oattr.record(put) if h.oattr is not None else {}

        meshes = []
        for t in self.tracks:
            if t.geom is None or not t.shown.any():
                continue
            o, g = t.ob, t.geom
            attrs, tris = _write_geom(blob, g)
            rec = {"name": o.name, "kind": t.kind,
                   "materials": [s.material.name if s.material else None for s in o.material_slots] or [None],
                   "visible": _visible(o), "aspect": o.get("mk_aspect"), "object": _object_rec(o),
                   "object_attrs": oattrs(t), "instance": None,
                   "shown": shown(t.shown), "matrix": matrix(t.M), "armature": None, "dqs": False,
                   "vertices": int(len(g["src"])), "attributes": attrs, "triangles": tris, "slots": g["slots"],
                   "morphs": None, "bend": None, "by_frame": None}
            if t.kind == "skinned":
                if t.arm.name not in arm_names:
                    raise RuntimeError(f"bake: lit: {o.name}'s armature {t.arm.name} has no joints in the bake")
                rec["armature"] = t.arm.name
                J, W = g["skin"]
                for i, (jk, wk) in enumerate((("joints", "weights"), ("joints2", "weights2"))[:J.shape[1] // 4]):
                    rec["attributes"][jk] = {"offset": blob.put(J[:, 4 * i:4 * i + 4], "u16"), "comp": 4, "type": "u16"}
                    rec["attributes"][wk] = {"offset": blob.put(W[:, 4 * i:4 * i + 4], "f32"), "comp": 4, "type": "f32"}
            if g.get("morphs") and g["morphs"][0]:
                names, idx, d = g["morphs"]
                vals = t.keys[:, idx] * np.array([not o.data.shape_keys.key_blocks[i].mute for i in idx], np.float32)
                rec["morphs"] = {"names": names, "deltas": blob.put(d, "f32"), "values": put(vals)}
            if g.get("bend") is not None:
                b = g["bend"]
                rec["attributes"]["fan"] = {"offset": blob.put(b["fan"], "u32"), "comp": 1, "type": "u32"}
                rec["attributes"]["bend"] = {"offset": blob.put(b["alpha_beta"], "f32"), "comp": 2, "type": "f32"}
                rec["bend"] = {"fans": len(b["center"]), **{k: blob.put(b[k], "u32") for k in ("center", "ref", "other")},
                               **{k: {"count": len(b[k][0]) - 1, "offsets": blob.put(b[k][0], "u32"),
                                      "items": blob.put(b[k][1], "u32"), "size": len(b[k][1])}
                                  for k in ("corners", "edges", "faces")}}
            t.rec = rec
            meshes.append(rec)
        by_source = {}
        for key in sorted(self.inst):
            r = self.inst[key]
            src = self.sources.get(r.of)
            shows = r.seen & self.shown.get(r.by, np.ones(n, bool))
            if not shows.any():
                continue
            if isinstance(src, str):
                self.left.setdefault(r.by, (src, r.by))
                continue
            if r.of not in by_source:
                by_source[r.of] = _write_geom(blob, src)
            o = bpy.data.objects[r.of]
            M = r.M.copy()
            M[~r.seen] = M[np.argmax(r.seen)]
            attrs, tris = by_source[r.of]
            meshes.append({"name": f"{r.by}/{r.of}/{'.'.join(str(i) for i in r.pid)}", "kind": "static",
                           "materials": [s.material.name if s.material else None for s in o.material_slots] or [None],
                           "visible": _visible(o), "aspect": o.get("mk_aspect"), "object": _object_rec(o),
                           "object_attrs": oattrs(r),
                           "instance": {"by": r.by, "of": r.of, "id": r.pid}, "shown": shown(shows), "matrix": matrix(M),
                           "armature": None, "dqs": False, "vertices": int(len(src["src"])), "attributes": attrs,
                           "triangles": tris, "slots": src["slots"], "morphs": None, "bend": None, "by_frame": None})
        for fx in self.flex:
            if fx.why is not None or not fx.geoms:
                if fx.why is not None:
                    self.left[fx.ob.name] = (fx.why, fx.ob.name)
                fx.discard()
                continue
            base = blob.append(fx.blob.path)
            fx.discard()
            for gm in fx.geoms:                             # their offsets are into its own part: now into lit.bin
                gm["triangles"]["offset"] += base
                for spec in gm["attributes"].values():
                    spec["offset"] += base
            o = fx.ob
            meshes.append({"name": o.name, "kind": "by_frame", "materials": fx.mats or [None], "visible": _visible(o),
                           "aspect": o.get("mk_aspect"), "object": _object_rec(o), "object_attrs": oattrs(fx),
                           "instance": None,
                           "shown": shown(fx.frames >= 0), "matrix": matrix(fx.matrices()), "armature": None,
                           "dqs": False, "why": fx.why_flex, "vertices": None, "attributes": None, "triangles": None,
                           "slots": None, "morphs": None, "bend": None,
                           "by_frame": {"geoms": fx.geoms, "frames": [int(f) for f in fx.frames]}})
        screen, screen_left = [], []
        for sw in self.screen:
            o = sw.ob
            show = o.get("mk_show")
            show = [int(show[0]), int(show[1])] if show else None
            if sw.why is not None or not sw.word.geoms:
                if sw.why is not None:
                    screen_left.append({"name": o.name, "why": sw.why, "aspect": o.get("mk_aspect"), "show": show})
                sw.discard()
                continue
            base = blob.append(sw.path)
            sw.bytes = os.path.getsize(sw.path)
            sw.discard()
            sw.rec = {"name": o.name, "kind": "by_frame", "materials": sw.mats or [None], "visible": _visible(o),
                      "aspect": o.get("mk_aspect"), "object": _object_rec(o), "instance": None,
                      "shown": shown(sw.word.frames >= 0), "matrix": matrix(np.eye(4)[None]), "armature": None,
                      "dqs": False, "why": "screen type", "vertices": None, "attributes": None, "triangles": None,
                      "slots": None, "morphs": None, "bend": None, "by_frame": sw.word.record(base), "show": show,
                      "knockout": "mk_knockout" in o.keys()}
            screen.append(sw.rec)
        for rec in meshes + screen:                         # the object whose look record selects it
            rec["record"] = rec["instance"]["by"] if rec["instance"] else rec["name"]
        records = {}
        for rec in meshes + screen:
            name = rec["record"]
            if name not in looks and name not in records:
                r = ST.record(bpy.data.objects[name])
                records[name] = {"collections": list(r.collections), "props": sorted(r.props)}
        points = {}
        for key, p in sorted(self.points.items()):
            runs = []
            for k in sorted(p["shots"]):
                if runs and runs[-1][1] == k and runs[-1][2] == p["shots"][k]:
                    runs[-1][1] = k + 1
                else:
                    runs.append([k, k + 1, p["shots"][k]])
            points[key] = {"expr": p["expr"], "offset": put(p["xyz"]), "frames": runs,
                           "errors": [{"error": e, "frames": fs} for e, fs in p["errors"].items()]}
        if self.screen_doc is not None:
            self.screen_doc.update(meshes=screen, left_out=screen_left)
        blob.close()
        lights = []
        for o, rows in self.lights:
            if not rows[:, 21].any():
                continue
            L = o.data
            still = np.allclose(rows, rows[:1])
            lights.append({"name": o.name, "type": L.type, "shape": L.shape if L.type == "AREA" else None,
                           "use_shadow": L.use_shadow, "visible": _visible(o), "aspect": o.get("mk_aspect"),
                           "props": _props(L, skip=_skip("id") | {"node_tree"}), "frames": 1 if still else n,
                           "offset": put(rows[:1] if still else rows)})
        cameras = {c: {"type": bpy.data.objects[c].data.type, "offset": put(self.cam_rows[c])} for c in self.cams}
        owners = {"material": self.mat_docs, "group": self.group_docs}
        for kind, owner, node, s, io, vals in self.chans:
            doc = (self.world_doc if self.world_doc and self.world_doc["name"] == owner else None) if kind == "world" \
                else owners[kind].get(owner)
            if doc is None:
                continue
            still = np.allclose(vals, vals[:1])
            doc["anim"].append({"node": node, "socket": s.identifier, "io": io, "size": int(vals.shape[1]),
                                "offset": put(vals[:1] if still else vals), "frames": 1 if still else n})
        left = [{"name": name, "why": why, "shows": _ranges(self.shown.get(of, np.zeros(n, bool)))}
                for name, (why, of) in sorted(self.left.items())]
        sc = self.sc
        self.docs = {"file": "lit.bin", "view": self.view,
                     "film": {"filter_size": sc.render.filter_size, "transparent": sc.render.film_transparent},
                     "render": _props(sc.render, skip={"filepath"}), "eevee": _props(sc.eevee),
                     "world": self.world_doc, "lights": lights, "materials": self.mat_docs, "groups": self.group_docs,
                     "images": self.image_docs, "cameras": cameras,
                     "objects": {name: matrix(M) for name, M in self.node_objects.items()},
                     "meshes": meshes, "left_out": left, "records": records, "screen": self.screen_doc,
                     "points": points, "points_error": self.points_error, "render_settings": self.render_changes,
                     "check": None}
        self.t_finish = time.time() - t0
        return self.docs

    def armatures(self):
        """Names of the armatures the skinned meshes need joints of."""
        return {t.arm.name for t in self.tracks if t.kind == "skinned" and t.shown.any()}

    def shows_any(self):
        return any(t.shown.any() for t in self.tracks) or any(fx.geoms for fx in self.flex) or bool(self.inst)

    # ------------------------------------------------------------------------------------------------- self-check
    def check(self, armatures):
        """Rebuild every mesh baked whole from lit.bin and bake.bin on the check frames and measure it against Blender's
        evaluated mesh: {"frames": [...], "meshes": {name: {"mm", "normal_deg": [max, p99, p99.9]}}, "worst_mm"}; a
        mesh that bends has its normals rebuilt as mkmmd.core.bend.normals does (and `skinned_normal_deg`: turned with the
        joints instead, for comparison)."""
        t0 = time.time()
        lit = np.fromfile(os.path.join(self.part, "lit.bin"), np.uint8)
        flt = np.fromfile(os.path.join(self.part, "bake.bin"), "<f4")
        arms = {a["name"]: a for a in armatures}

        def arr(at, n, dt, comp=1):
            a = np.frombuffer(lit, np.dtype(DTYPES[dt]), n * comp, at)
            return a.reshape(n, comp) if comp > 1 else a

        def attr(spec, nv):
            return arr(spec["offset"], nv, spec["type"], spec["comp"]).astype(float)

        def degrees(n, tn):
            n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            return np.degrees(np.arccos(np.clip(np.einsum("ci,ci->c", n[t.geom["corner"]], tn), -1.0, 1.0)))

        def stats(ang):
            ang = np.nan_to_num(np.concatenate(ang), nan=180.0)
            return [round(float(ang.max()), 3), round(float(np.percentile(ang, 99)), 3),
                    round(float(np.percentile(ang, 99.9)), 3)]
        out = {}
        for t in self.tracks:
            rec = t.rec
            if rec is None or not t.truth:
                continue
            nv, a = rec["vertices"], rec["attributes"]
            pos0, nor0 = attr(a["position"], nv), attr(a["normal"], nv)
            tris = arr(rec["triangles"]["offset"], rec["triangles"]["count"], "u32", 3).astype(np.int64)
            b = rec["bend"]
            if b is not None:
                nf = b["fans"]
                ab = attr(a["bend"], nv)
                bend = {"fan": attr(a["fan"], nv).astype(np.int64), "alpha": ab[:, 0], "beta": ab[:, 1],
                        **{k: arr(b[k], nf, "u32").astype(np.int64) for k in ("center", "ref", "other")},
                        **{k: (arr(b[k]["offsets"], b[k]["count"] + 1, "u32").astype(np.int64),
                               arr(b[k]["items"], b[k]["size"], "u32").astype(np.int64))
                           for k in ("corners", "edges", "faces")}}
            worst, angles, skinned = 0.0, [], []
            for k, (tp, tn) in sorted(t.truth.items()):
                pos = pos0.copy()
                if rec["morphs"]:
                    m = rec["morphs"]
                    nk = len(m["names"])
                    d = np.frombuffer(lit, "<f4", nk * nv * 3, m["deltas"]).reshape(nk, nv, 3)
                    pos += np.tensordot(flt[m["values"] + nk * k:][:nk].astype(float), d, 1)
                mo = rec["matrix"]["offset"] + 12 * (k if rec["matrix"]["frames"] > 1 else 0)
                M = flt[mo:mo + 12].astype(float).reshape(4, 3).T
                p = pos @ M[:, :3].T + M[:, 3]
                A = np.broadcast_to(M[:, :3], (nv, 3, 3))
                if rec["kind"] == "skinned":
                    arm = arms[rec["armature"]]
                    nb = len(arm["bones"])
                    J = flt[arm["offset"] + nb * 12 * k:][:nb * 12].astype(float).reshape(nb, 4, 3).transpose(0, 2, 1)
                    JI = np.concatenate([attr(a[j], nv) for j in ("joints", "joints2") if j in a], 1).astype(int)
                    W = np.concatenate([attr(a[w], nv) for w in ("weights", "weights2") if w in a], 1)
                    S = np.einsum("vk,vkij->vij", W, J[JI])
                    S[W.sum(1) <= 0.0] = np.eye(4)[:3]
                    p = np.einsum("vij,vj->vi", S[:, :, :3], p) + S[:, :, 3]
                    A = np.einsum("vij,jk->vik", S[:, :, :3], M[:, :3])
                worst = max(worst, float(np.linalg.norm(p - tp[t.geom["src"]], axis=1).max()))
                turned = np.einsum("vji,vj->vi", np.linalg.inv(A), nor0)    # inverse transpose
                if b is not None:
                    angles.append(degrees(BD.normals(p, tris, bend), tn))
                    skinned.append(degrees(turned, tn))
                else:
                    angles.append(degrees(turned, tn))
            out[rec["name"]] = {"mm": round(worst * 1000.0, 4), "normal_deg": stats(angles)}
            if skinned:
                out[rec["name"]]["skinned_normal_deg"] = stats(skinned)
        screen = {}
        for sw in self.screen:                             # screen type: its check frames decoded from lit.bin
            if sw.rec is None:
                continue
            h = self._screen_px(sw)
            worst = worst_n = worst_a = 0.0
            problems = []
            for k, truth in sorted(sw.truth.items()):
                dec = FG.decode(lit, sw.rec["by_frame"], k)
                if dec is None:
                    problems.append(f"frame {self.frame0 + k}: hidden in the bake")
                    continue
                idx, tp, tn, ta, solid = truth
                worst = max(worst, float(np.linalg.norm(dec["position"][idx] - tp, axis=1).max()))
                cos = np.einsum("vi,vi->v", dec["normal"][idx].astype(float), tn.astype(float))[solid]
                worst_n = max(worst_n, float(np.degrees(np.arccos(np.clip(cos.min(initial=1.0), -1.0, 1.0)))))
                for key, a in ta.items():
                    got = dec["attrs"].get(key)
                    worst_a = max(worst_a, float(np.abs(got[idx] - a.reshape(len(idx), -1)).max())
                                  if got is not None else float("inf"))
            st = sw.word.stats
            screen[sw.ob.name] = {"frames": [self.frame0 + k for k in sorted(sw.truth)], "px": round(worst * h, 5),
                                  "normal_deg": round(worst_n, 4), "attr": worst_a, "problems": problems,
                                  "fit": {"frames": st["frames"], "word": st["word"], "islands": st["islands"],
                                          "bent": st["bent"], "geoms": st["geoms"], "refit": st["refit"],
                                          "projective": st["projective"], "px": round(st["err"] * h, 5)},
                                  "bytes": sw.bytes}
        self.t_check = time.time() - t0
        return {"frames": self.check_k, "fit_frames": self.fit_k, "meshes": out, "screen": screen,
                "worst_mm": max([v["mm"] for v in out.values()], default=0.0),
                "screen_worst_px": max([v["px"] for v in screen.values()], default=None)}

    def _screen_px(self, sw):
        """Pixels a world unit of screen type spans in the word's output (one frame height; 1080 without one)."""
        cam = ((self.screen_doc or {}).get("cameras") or {}).get(sw.ob.get("mk_aspect"))
        return float(cam["size"][1]) if cam else 1080.0

    def report(self, folder):
        """The bake report's `lit` entry (`folder`: where the bake now is)."""
        d = self.docs or {}
        kinds = {}
        for m in d.get("meshes", []):
            kind = "instance" if m["instance"] else m["kind"]
            kinds[kind] = kinds.get(kind, 0) + 1
        chk = d.get("check") or {}
        scr = chk.get("screen") or {}
        fits = [v["fit"] for v in scr.values()]
        return {"meshes": kinds, "lights": len(d.get("lights", [])), "materials": len(d.get("materials", {})),
                "images": len(d.get("images", {})), "left_out": {e["name"]: e["why"] for e in d.get("left_out", [])},
                "lit_mb": round(os.path.getsize(os.path.join(folder, "lit.bin")) / 1e6, 1),
                "check_mm": {k: v["mm"] for k, v in (chk.get("meshes") or {}).items()},
                "check_worst_mm": chk.get("worst_mm"),
                "screen": {"meshes": len((d.get("screen") or {}).get("meshes", [])),
                           "left_out": {e["name"]: e["why"] for e in (d.get("screen") or {}).get("left_out", [])},
                           "mb": round(sum(v["bytes"] for v in scr.values()) / 1e6, 2),
                           "frames": sum(f["frames"] for f in fits),
                           "poses": {kind: sum(f[kind] for f in fits) for kind in ("word", "islands", "bent")},
                           "geoms": sum(f["geoms"] for f in fits), "refits": sum(f["refit"] for f in fits),
                           "projective_maps": sum(f["projective"] for f in fits),
                           "fit_worst_px": max([f["px"] for f in fits], default=None),
                           "check_worst_px": chk.get("screen_worst_px"),
                           "check_worst_normal_deg": max([v["normal_deg"] for v in scr.values()], default=None),
                           "check_worst_attr": max([v["attr"] for v in scr.values()], default=None),
                           "problems": {k: v["problems"] for k, v in scr.items() if v["problems"]},
                           "cache": {"hits": sum(sw.cached for sw in self.screen),
                                     "baked": sum(sw.key is not None and not sw.cached for sw in self.screen),
                                     "uncacheable": sum(sw.key is None for sw in self.screen)},
                           "seconds": {"keys": round(self.t_screen_keys, 2),
                                       "evaluate": round(sum(sw.t_eval for sw in self.screen), 2),
                                       "fit": round(sum(sw.t_fit for sw in self.screen), 2)}},
                "points": {k: len(p["errors"]) for k, p in (d.get("points") or {}).items()},
                "render_settings": d.get("render_settings") or {},
                "seconds": {"lut": round(self.t_lut, 2), "frames": round(self.t_sample, 2),
                            "frames_by_part": {k: round(v, 2) for k, v in self.t_parts.items()},
                            "capture": round(self.t_capture, 2), "write": round(self.t_finish, 2),
                            "check": round(self.t_check, 2)}}
