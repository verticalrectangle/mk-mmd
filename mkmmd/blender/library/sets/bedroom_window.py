"""The window of the bedroom_80s set: painted frame with mullions, casing, sill board and apron, the glass, and the
venetian blind (a geometry-nodes object driven by the set's `blinds` and `slat_angle` properties). A helper module: it
registers no builder.

The blind hangs in front of the wall (outside mount): headrail above the opening, slats in the plane w = BLIND_W, tilt
cord on the left, lift cord with a tassel on the right. Its slats are instances of one hidden slat mesh placed by a node
tree: slat i sits at max(its hanging place, its place in the pile on the bottom rail), see bedroom_maths.blind_levels,
which the tree repeats node for node."""
import math

import bmesh
import bpy
from mathutils import Vector

from ..props.cafe_kit import bm_append, bm_tube, drive
from . import bedroom_maths as BM
from . import bedroom_shape as SH
from .bedroom_parts import _face, wbox
from .cafe_nodes import hide_from_rays

RING = 0.055                 # width of the frame's outer members
BAR = 0.035                  # muntin width
MEET = 0.05                  # meeting rail width


def frame_boxes(cfg):
    """Boxes (wall frame u0, u1, v0, v1, w0, w1) of the window's lining, meeting rails and muntins."""
    w = cfg["window"]
    x0, x1, z0, z1, rows, cols = w["x0"], w["x1"], w["z0"], w["z1"], w["rows"], w["cols"]
    W0, W1 = BM.FRAME_W0, BM.FRAME_W1
    bottom = z0 + 0.05
    boxes = [(x0, x0 + RING, z0, z1, W0, W1), (x1 - RING, x1, z0, z1, W0, W1),
             (x0 + RING, x1 - RING, z0, bottom, W0, W1), (x0 + RING, x1 - RING, z1 - RING, z1, W0, W1)]
    rails = [z0 + w["height"] * k / rows for k in range(1, rows)]
    boxes += [(x0 + RING, x1 - RING, z - MEET / 2, z + MEET / 2, W0 - 0.012, W1) for z in rails]
    lows = [bottom] + [z + MEET / 2 for z in rails]
    highs = [z - MEET / 2 for z in rails] + [z1 - RING]
    for lo, hi in zip(lows, highs):
        for k in range(1, cols):
            xc = x0 + w["width"] * k / cols
            boxes.append((xc - BAR / 2, xc + BAR / 2, lo, hi, W0 + 0.010, W1 - 0.010))
    return boxes


SILL_THICK = 0.035           # the sill board (stool)


def trim_boxes(cfg):
    """Casing, head and apron boards in the wall frame (their backs sunk into the wall); the sill board is a sweep."""
    w = cfg["window"]
    CW, CT, B = BM.CASING_W, BM.CASING_T, BM.BURY
    x0, x1, z0, z1 = w["x0"], w["x1"], w["z0"], w["z1"]
    return [(x0 - CW, x0, z0, z1 + CW, -CT, B), (x1, x1 + CW, z0, z1 + CW, -CT, B),
            (x0 - CW - 0.012, x1 + CW + 0.012, z1, z1 + CW, -CT - 0.004, B),
            (x0 - CW, x1 + CW, z0 - SILL_THICK - 0.065, z0 - SILL_THICK, -CT, B)]


def sill_bm(cfg):
    """The sill board: a rounded section (nosing 0.06 m into the room, back flush with the frame) swept along the wall,
    from the right to the left so that the room is on the sweep's left."""
    w = cfg["window"]
    ext = BM.CASING_W + 0.022
    profile = [(d, z + w["z0"]) for d, z in BM.sill_profile()]
    bm = bmesh.new()
    SH.sweep_bm(bm, [(w["x1"] + ext, cfg["y1"]), (w["x0"] - ext, cfg["y1"])], profile)
    SH.bevel_hard(bm, 0.0012, 2)                     # the rims of the two ends
    return bm


def head_bm(cfg, lay):
    """The headrail: a rounded channel section swept along the window wall (see bedroom_maths.HEAD_PROFILE)."""
    half = lay["width"] / 2 + 0.012
    profile = [(d, z + lay["top"]) for d, z in BM.moulding(BM.HEAD_PROFILE, BM.HEAD_FILLET, 4)]
    bm = bmesh.new()
    SH.sweep_bm(bm, [(lay["x"] + half, cfg["y1"]), (lay["x"] - half, cfg["y1"])], profile)
    SH.bevel_hard(bm, 0.0012, 2)
    return bm


def slat_mesh(name, length, chord, sag=0.0022, thick=0.0004, seg=6):
    """One slat: a shallow cove (middle lower than the edges) of the given chord, centred on the origin, running along
    x; smooth shaded across the arc."""
    bm = bmesh.new()
    ys = [(-0.5 + i / seg) * chord for i in range(seg + 1)]
    zt = [sag * ((2 * y / chord) ** 2 - 0.5) for y in ys]
    hx = length / 2
    for i in range(seg):
        pa, pb = (ys[i], zt[i]), (ys[i + 1], zt[i + 1])
        up = (0.0, 0.0, 1.0)
        f = _face(bm, [(-hx, pa[0], pa[1]), (hx, pa[0], pa[1]), (hx, pb[0], pb[1]), (-hx, pb[0], pb[1])], up)
        f.smooth = True
        f = _face(bm, [(-hx, pa[0], pa[1] - thick), (hx, pa[0], pa[1] - thick), (hx, pb[0], pb[1] - thick),
                       (-hx, pb[0], pb[1] - thick)], (0.0, 0.0, -1.0))
        f.smooth = True
    for y, z, n in ((ys[0], zt[0], (0, -1, 0)), (ys[-1], zt[-1], (0, 1, 0))):
        _face(bm, [(-hx, y, z), (hx, y, z), (hx, y, z - thick), (-hx, y, z - thick)], n)
    for x, n in ((-hx, (-1, 0, 0)), (hx, (1, 0, 0))):
        pts = [(x, y, z) for y, z in zip(ys, zt)] + [(x, y, z - thick) for y, z in zip(ys[::-1], zt[::-1])]
        _face(bm, pts, n)
    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


class _GN:
    """A few helpers over a geometry node tree."""

    def __init__(self, nt):
        self.nt = nt

    def node(self, idname, **props):
        n = self.nt.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def put(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        elif v is not None:
            sock.default_value = v

    def math(self, op, a, b=None, c=None):
        n = self.node("ShaderNodeMath", operation=op)
        self.put(n.inputs[0], a)
        self.put(n.inputs[1], b)
        self.put(n.inputs[2], c)
        return n.outputs[0]

    def xyz(self, x=0.0, y=0.0, z=0.0):
        n = self.node("ShaderNodeCombineXYZ")
        self.put(n.inputs[0], x)
        self.put(n.inputs[1], y)
        self.put(n.inputs[2], z)
        return n.outputs[0]

    def transform(self, geo, t=None, s=None, r=None):
        n = self.node("GeometryNodeTransform")
        self.put(n.inputs["Geometry"], geo)
        self.put(n.inputs["Translation"], t)
        self.put(n.inputs["Rotation"], r)
        self.put(n.inputs["Scale"], s)
        return n.outputs[0]

    def material(self, geo, mat):
        n = self.node("GeometryNodeSetMaterial")
        self.put(n.inputs["Geometry"], geo)
        n.inputs["Material"].default_value = mat
        return n.outputs[0]

    def join(self, *geos):
        n = self.node("GeometryNodeJoinGeometry")
        for g in geos:
            self.nt.links.new(g, n.inputs[0])
        return n.outputs[0]

    def cylinder(self, radius, depth, vertices, fill="NGON"):
        n = self.node("GeometryNodeMeshCylinder", fill_type=fill)
        self.put(n.inputs["Vertices"], vertices)
        self.put(n.inputs["Radius"], radius)
        self.put(n.inputs["Depth"], depth)
        return n.outputs["Mesh"]

    def sphere(self, radius, segments, rings):
        n = self.node("GeometryNodeMeshUVSphere")
        self.put(n.inputs["Segments"], segments)
        self.put(n.inputs["Rings"], rings)
        self.put(n.inputs["Radius"], radius)
        return n.outputs["Mesh"]

    def smooth(self, geo):
        n = self.node("GeometryNodeSetShadeSmooth")
        self.put(n.inputs["Geometry"], geo)
        self.put(n.inputs["Shade Smooth"], True)
        return n.outputs[0]


def blinds_tree(name, lay, slat_obj, mats):
    """The blind's node tree: inputs Raise (0..1) and Angle (radians); output the slats, the bottom rail, the ladder
    strings, the lift cord and its tassel, in the frame of the headrail's bottom centre (z down)."""
    old = bpy.data.node_groups.get(f"{name}_blinds")
    if old is not None:
        bpy.data.node_groups.remove(old)
    nt = bpy.data.node_groups.new(f"{name}_blinds", "GeometryNodeTree")
    nt.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    raise_in = nt.interface.new_socket("Raise", in_out="INPUT", socket_type="NodeSocketFloat")
    angle_in = nt.interface.new_socket("Angle", in_out="INPUT", socket_type="NodeSocketFloat")
    g = _GN(nt)
    gin, gout = g.node("NodeGroupInput"), g.node("NodeGroupOutput")
    n, pf, pm, first = lay["n"], lay["pitch"], lay["pitch_min"], lay["first"]
    width, chord = lay["width"], lay["chord"]
    z_low = -first - (n - 1) * pf - pm
    z_high = -first - n * pm
    rail = g.math("MULTIPLY_ADD", gin.outputs["Raise"], z_high - z_low, z_low)            # top of the bottom rail

    # slats: points on a line, each at the higher of its hanging place and its place in the pile, tilted by Angle
    line = g.node("GeometryNodeMeshLine", mode="OFFSET", count_mode="TOTAL")
    g.put(line.inputs["Count"], n)
    g.put(line.inputs["Offset"], (0.0, 0.0, -1.0))
    i = g.math("ADD", g.node("GeometryNodeInputIndex").outputs[0], 0.0)
    hang = g.math("SUBTRACT", -first, g.math("MULTIPLY", i, pf))
    pile = g.math("ADD", rail, g.math("MULTIPLY", g.math("SUBTRACT", float(n), i), pm))
    setpos = g.node("GeometryNodeSetPosition")
    g.put(setpos.inputs["Geometry"], line.outputs[0])
    g.put(setpos.inputs["Position"], g.xyz(0.0, 0.0, g.math("MAXIMUM", hang, pile)))
    info = g.node("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    info.inputs["Object"].default_value = slat_obj
    info.inputs["As Instance"].default_value = True
    tilt = g.node("FunctionNodeAxisAngleToRotation")
    g.put(tilt.inputs["Axis"], (1.0, 0.0, 0.0))
    g.put(tilt.inputs["Angle"], gin.outputs["Angle"])
    iop = g.node("GeometryNodeInstanceOnPoints")
    g.put(iop.inputs["Points"], setpos.outputs[0])
    g.put(iop.inputs["Instance"], info.outputs["Geometry"])
    g.put(iop.inputs["Rotation"], tilt.outputs[0])
    slats = g.node("GeometryNodeRealizeInstances")
    g.put(slats.inputs[0], iop.outputs[0])
    metal = g.material(slats.outputs[0], mats["slat"])

    # bottom rail: a flat tube (20 x 12 mm) with rounded ends, no sharp edge
    rail_z = g.math("SUBTRACT", rail, lay["rail_t"] / 2)
    tube = g.transform(g.cylinder(0.5, width, 16, "NONE"), g.xyz(0.0, 0.0, rail_z), (0.012, 0.020, 1.0),
                       (0.0, math.pi / 2, 0.0))
    ends = [g.transform(g.sphere(0.5, 16, 8), g.xyz(sx * width / 2, 0.0, rail_z), (0.012, 0.020, 0.012))
            for sx in (-1.0, 1.0)]
    rail_geo = g.material(g.smooth(g.join(tube, *ends)), mats["slat"])

    # ladder strings: a front and a back string (round threads) at three places along the blind
    length = g.math("SUBTRACT", g.math("MULTIPLY", rail, -1.0), 0.002)
    y_off = g.math("MULTIPLY", g.math("COSINE", gin.outputs["Angle"]), chord / 2 - 0.0025)
    strings = [g.transform(g.cylinder(0.0007, 1.0, 6, "NONE"),
                           g.xyz(0.0, g.math("MULTIPLY", y_off, sign), g.math("MULTIPLY", length, -0.5)),
                           g.xyz(1.0, 1.0, length)) for sign in (1.0, -1.0)]
    pair = g.join(*strings)
    xs = g.node("GeometryNodeMeshLine", mode="OFFSET", count_mode="TOTAL")
    g.put(xs.inputs["Count"], 3)
    g.put(xs.inputs["Start Location"], (-(width / 2 - 0.10), 0.0, -0.002))
    g.put(xs.inputs["Offset"], (width / 2 - 0.10, 0.0, 0.0))
    ladders = g.node("GeometryNodeInstanceOnPoints")
    g.put(ladders.inputs["Points"], xs.outputs[0])
    g.put(ladders.inputs["Instance"], pair)
    ladder_geo = g.material(_realize(g, ladders.outputs[0]), mats["string"])

    # lift cord and tassel on the right: the cord is pulled out as the blind goes up
    xc, yc = width / 2 - 0.03, -0.017                      # 1.7 cm in front of the slats' plane
    cl = g.math("MULTIPLY_ADD", gin.outputs["Raise"], 0.7, 0.45)
    cord = g.transform(g.cylinder(0.0016, cl, 6, "NONE"),
                       g.xyz(xc, yc, g.math("SUBTRACT", g.math("MULTIPLY", cl, -0.5), 0.002)))
    tassel = g.transform(g.sphere(0.5, 12, 8), g.xyz(xc, yc, g.math("SUBTRACT", g.math("MULTIPLY", cl, -1.0), 0.0215)),
                         (0.020, 0.020, 0.045))
    cord_geo = g.material(g.smooth(g.join(cord, tassel)), mats["cord"])

    g.put(gout.inputs[0], g.join(metal, rail_geo, ladder_geo, cord_geo))
    return nt, raise_in.identifier, angle_in.identifier


def _realize(g, geo):
    n = g.node("GeometryNodeRealizeInstances")
    g.put(n.inputs[0], geo)
    return n.outputs[0]


def _glass(R, cfg, fr, mats):
    """The pane (one quad over the opening, no shadow) and the planar probe on it: what the faint reflection shows is
    the room, not the sky."""
    w = cfg["window"]
    gw = (BM.FRAME_W0 + BM.FRAME_W1) / 2
    corners = ((w["x0"], w["z0"]), (w["x1"], w["z0"]), (w["x1"], w["z1"]), (w["x0"], w["z1"]))
    verts = [tuple(BM.to_world(fr, u, v, gw)) for u, v in corners]
    glass = R.mesh_obj("window_glass", verts, [(0, 1, 2, 3)], "window", [mats["glass"]],
                       uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
    hide_from_rays(glass, shadow=False)
    probe = bpy.data.lightprobes.new(R.oname("window_mirror"), "PLANE")
    at = BM.to_world(fr, w["x"], (w["z0"] + w["z1"]) / 2, gw - 0.004)
    mirror = R.obj("window_mirror", probe, "window", loc=tuple(at), rot=(math.pi / 2, 0.0, 0.0),
                   scale=(w["width"] / 2, w["height"] / 2, 1.0))
    return glass, mirror


def _headrail(R, cfg, fr, lay, mats):
    """The static part of the blind: the headrail (a rounded channel), two brackets, the tilt wand with its knob."""
    bx, top, wy = lay["x"], lay["top"], BM.BLIND_W
    half = lay["width"] / 2 + 0.012
    bm = head_bm(cfg, lay)
    brackets = SH.box_bm([([wbox(fr, ub - 0.012, ub + 0.012, top + 0.006, top + BM.HEAD_H - 0.006, wy + 0.015, BM.BURY)
                            for ub in (bx - half + 0.12, bx + half - 0.12)], 0.0015, 3)])
    bm_append(bm, brackets)
    brackets.free()
    wu, ww = bx - lay["width"] / 2 + 0.03, wy - 0.0175
    end = top - 0.92

    def at(v):
        return Vector(BM.to_world(fr, wu, v, ww))
    bm_tube(bm, [at(top), at(end)], 0.0032, sides=8, mat=0, cap="none", smooth=True, uv=False)
    bm_tube(bm, [at(end), at(end - 0.045)], 0.0052, sides=10, mat=0, cap="round", smooth=True, uv=False)
    return SH.to_object(R, "window_blind_head", bm, [mats["slat"]], "window")


def _blinds(R, cfg, fr, lay, mats):
    """The slats' source mesh (hidden: only the node tree reads it) and the node-tree object, driven by the root's
    `blinds` and `slat_angle`."""
    bx, top = lay["x"], lay["top"]
    slat = slat_mesh(R.oname("window_blind_slat"), lay["width"] - 0.004, lay["chord"])
    src = R.obj("window_blind_slat", slat, "window")
    src.hide_render = True
    src.hide_viewport = True
    nt, rid, aid = blinds_tree(R.name, lay, src, mats)
    host = R.obj("window_blinds", bpy.data.meshes.new(R.oname("window_blinds")), "window",
                 loc=tuple(BM.to_world(fr, bx, top, BM.BLIND_W)))
    mod = host.modifiers.new("Blinds", "NODES")
    mod.node_group = nt
    mod[rid] = float(cfg["params"]["blinds"])
    mod[aid] = math.radians(cfg["params"]["slat_angle"])
    drive(host, f'modifiers["Blinds"]["{rid}"]', "p", var=("p", R.root, '["blinds"]'))
    drive(host, f'modifiers["Blinds"]["{aid}"]', f"p * {math.pi / 180.0!r}", var=("p", R.root, '["slat_angle"]'))
    return host, src


def build_window(R, cfg, mats):
    """Objects: window_frame, window_trim, window_glass, window_mirror (planar probe), window_blind_head, window_blinds
    (node tree) and window_blind_slat (hidden source mesh of the slats). Returns {name without the set's prefix:
    object}."""
    if "back" in cfg["open"]:
        return {}
    fr = BM.wall_frame(cfg, "back")
    lay = BM.blind_layout(cfg)
    trim = SH.box_bm([([wbox(fr, *b) for b in trim_boxes(cfg)], 0.003, 3)])
    sill = sill_bm(cfg)
    bm_append(trim, sill)
    sill.free()
    out = {"window_frame": SH.boxes_obj(R, "window_frame", [([wbox(fr, *b) for b in frame_boxes(cfg)], 0.003, 3)],
                                        [mats["trim"]], "window"),
           "window_trim": SH.to_object(R, "window_trim", trim, [mats["trim"]], "window")}
    out["window_glass"], out["window_mirror"] = _glass(R, cfg, fr, mats)
    out["window_blind_head"] = _headrail(R, cfg, fr, lay, mats)
    out["window_blinds"], out["window_blind_slat"] = _blinds(R, cfg, fr, lay, mats)
    return out
