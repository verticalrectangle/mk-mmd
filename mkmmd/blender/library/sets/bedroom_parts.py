"""Geometry of the bedroom_80s set that is not the window: the walls, floor and ceiling, skirting, picture rail, the
door, the wall fixtures, the neon strip and the ceiling rose. A helper module: it registers no builder.

Every function takes the shared `Ctx` (sets/cafe_nodes.py), the parsed spec `cfg` (bedroom_maths.parse) and the
materials dict, and returns the objects it made. Wall parts are placed in the wall's frame (u along the wall, v up, w
into the wall; bedroom_maths.wall_frame) and converted to boxes in set coordinates. Nothing has a sharp edge
(bedroom_shape): boxes are bevelled, long mouldings are swept from rounded sections."""
import bmesh
import bpy
from mathutils import Matrix, Vector

from ..props.cafe_kit import bm_append, bm_lathe, bm_tube
from . import bedroom_maths as BM
from . import bedroom_shape as SH
from .cafe_nodes import axis_mat

SLOT = {"room": 0, "reveal": 1, "outer": 2, "pocket": 3}       # material slots of a wall object


def _mesh(R, base, bm, mats, group="room", smooth=False):
    bm.normal_update()
    if smooth:
        for f in bm.faces:
            f.smooth = True
    me = bpy.data.meshes.new(R.oname(base))
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    return R.obj(base, me, group)


def wbox(fr, u0, u1, v0, v1, w0, w1):
    """A box given in a wall's frame as (x0, x1, y0, y1, z0, z1) in set coordinates."""
    a, b = BM.to_world(fr, u0, v0, w0), BM.to_world(fr, u1, v1, w1)
    return (min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1]), min(a[2], b[2]), max(a[2], b[2]))


def _face(bm, pts, normal, mat=0, uvs=None):
    """A polygon whose winding is turned to agree with `normal`. `uvs` maps UV layers to either one (u, v) per vertex
    (a list) or a single (u, v) for the whole polygon."""
    p = [Vector(q) for q in pts]
    nrm = Vector((0, 0, 0))
    for i in range(len(p)):
        nrm += p[i].cross(p[(i + 1) % len(p)])
    order = list(range(len(p)))
    if nrm.dot(Vector(normal)) < 0:
        order.reverse()
    f = bm.faces.new([bm.verts.new(p[i]) for i in order])
    f.material_index = mat
    if uvs:
        for lp, i in zip(f.loops, order):
            for layer, uv in uvs.items():
                lp[layer].uv = uv[i] if isinstance(uv, list) else uv
    return f


# ---------------------------------------------------------------------------------------------------- shell


def build_shell(R, cfg, mats):
    """{object name without the set's prefix: object} of the four walls (wall_back, wall_left, wall_right, wall_front),
    the ceiling, the roof-shadow plane and the floor (planks)."""
    out = {}
    for wall in BM.WALLS:
        if wall in cfg["open"]:
            continue
        V, faces, tags = BM.wall_mesh(cfg, wall)
        bm = bmesh.new()
        vs = [bm.verts.new((float(p[0]), float(p[1]), float(p[2]))) for p in V]
        for f, t in zip(faces, tags):
            bm.faces.new([vs[i] for i in f]).material_index = SLOT[t]
        SH.bevel_hard(bm, 0.003, 3)         # the arrises of the opening and the pocket, the hidden outer edges
        paper = mats["paper_x"] if wall in ("back", "front") else mats["paper_y"]
        reveal = mats["trim"] if wall == "right" else mats["reveal"]       # the door's lining: painted like its casing
        out[f"wall_{wall}"] = _mesh(R, f"wall_{wall}", bm, [paper, reveal, mats["outside"], mats["door_dark"]])
    x0, x1, y0, y1, H, T = cfg["x0"], cfg["x1"], cfg["y0"], cfg["y1"], cfg["H"], cfg["T"]
    if "ceiling" not in cfg["open"]:         # over the whole footprint: its rounded edges are inside the walls
        out["ceiling"] = SH.boxes_obj(R, "ceiling", [([(x0 - T, x1 + T, y0 - T, y1 + T, H, H + T)], 0.003, 3)],
                                      [mats["ceiling"], mats["outside"]],
                                      face_slot=lambda n: axis_mat(n, {"-z": 0, "any": 1}))
        out["roof_shadow"] = build_roof_shadow(R, cfg)
    if "floor" not in cfg["open"]:
        out["floor"] = build_floor(R, cfg, mats)
    return out


def build_roof_shadow(R, cfg):
    """`<name>_roof_shadow`: the roof as the lights see it. An invisible plane over the interior at the ceiling
    (camera, reflection and bounce rays pass through it, shadow rays do not), so hiding `<name>_ceiling` for a plan
    view takes the ceiling out of the picture but not out of the lighting: the sun does not pour over the wall tops
    and light the far floor with a crisp-edged patch, and the sky does not light the room from above.
    `open = ["ceiling"]` leaves the room open to the sky and builds no shadow plane."""
    x0, x1, y0, y1, H = cfg["x0"], cfg["x1"], cfg["y0"], cfg["y1"], cfg["H"]
    me = bpy.data.meshes.new(R.oname("roof_shadow"))
    z = H + 0.002
    me.from_pydata([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [], [(0, 1, 2, 3)])
    me.update()
    o = R.obj("roof_shadow", me, "room")
    o.visible_camera = o.visible_diffuse = o.visible_glossy = o.visible_transmission = False
    o.visible_volume_scatter = False
    o.visible_shadow = True
    return o


CHAMFER = 0.0008                 # the planks' arris: 0.8 mm at 45 degrees (varnished parquet is never razor-edged)


def build_floor(R, cfg, mats):
    """Parquet: every plank is a top face with its own UVs ("plank": 0..1 along and across, "rand": two random numbers),
    a 0.8 mm chamfer all round and four short sides down to the dark underlay, so the groove between planks is a real
    gap with soft edges. (A sliver too narrow for the chamfer keeps a square edge.)"""
    x0, x1, y0, y1, T = cfg["x0"], cfg["x1"], cfg["y0"], cfg["y1"], cfg["T"]
    planks = BM.parquet(x0, x1, y0, y1, kind=cfg["floor"], seed=cfg["seed"])
    bm = bmesh.new()
    uv_p = bm.loops.layers.uv.new("plank")
    uv_r = bm.loops.layers.uv.new("rand")
    depth, ch = 0.004, CHAMFER
    for pk in planks:
        poly = pk["poly"]
        c, a = pk["c"], pk["a"]
        b = (-a[1], a[0])

        def uvs(pts):
            return {uv_p: [((((x - c[0]) * a[0] + (y - c[1]) * a[1]) / pk["L"]) + 0.5,
                            (((x - c[0]) * b[0] + (y - c[1]) * b[1]) / pk["w"]) + 0.5) for x, y in pts],
                    uv_r: tuple(pk["r"])}
        n = len(poly)
        top = BM.inset_polygon(poly, ch)
        if top is not None and len(top) == n:
            _face(bm, [(x, y, 0.0) for x, y in top], (0, 0, 1), 0, uvs(top))
            for i in range(n):
                i2 = (i + 1) % n
                quad = [top[i], top[i2], poly[i2], poly[i]]
                (xa, ya), (xb, yb) = poly[i], poly[i2]
                _face(bm, [(*top[i], 0.0), (*top[i2], 0.0), (*poly[i2], -ch), (*poly[i], -ch)],
                      ((yb - ya), -(xb - xa), ch), 0, uvs(quad))
            z0 = -ch
        else:
            _face(bm, [(x, y, 0.0) for x, y in poly], (0, 0, 1), 0, uvs(poly))
            z0 = 0.0
        for i in range(n):
            (xa, ya), (xb, yb) = poly[i], poly[(i + 1) % n]
            _face(bm, [(xa, ya, z0), (xb, yb, z0), (xb, yb, -depth), (xa, ya, -depth)], (yb - ya, -(xb - xa), 0.0), 1)
    _face(bm, [(x0, y0, -depth), (x1, y0, -depth), (x1, y1, -depth), (x0, y1, -depth)], (0, 0, 1), 1)
    for p, q, nrm in (((x0, y0), (x1, y0), (0, -1, 0)), ((x1, y0), (x1, y1), (1, 0, 0)),
                      ((x1, y1), (x0, y1), (0, 1, 0)), ((x0, y1), (x0, y0), (-1, 0, 0))):
        _face(bm, [(p[0], p[1], -depth), (q[0], q[1], -depth), (q[0], q[1], -T), (p[0], p[1], -T)], nrm, 2)
    _face(bm, [(x0, y0, -T), (x1, y0, -T), (x1, y1, -T), (x0, y1, -T)], (0, 0, -1), 2)
    return _mesh(R, "floor", bm, [mats["parquet"], mats["groove"], mats["outside"]])


# ---------------------------------------------------------------------------------------------------- trim


def build_trim(R, cfg, mats):
    """Skirting and picture rail (sections swept round the room, mitred at the corners; the skirting stops at the door
    casing) and the casing of the door (bevelled boards)."""
    d = cfg["door"]
    CW, CT, BURY = BM.CASING_W, BM.CASING_T, BM.BURY
    out = {}
    gap = (d["y0"] - CW, d["y1"] + CW) if "right" not in cfg["open"] else None
    runs = BM.wall_runs(cfg, gap)
    if runs:
        out["skirting"] = SH.moulded_obj(R, "skirting", runs, BM.moulding(BM.SKIRT_PROFILE, BM.SKIRT_FILLET, 4),
                                         [mats["trim"]])
        rail = [(dd, z + cfg["rail_z"]) for dd, z in BM.moulding(BM.RAIL_PROFILE, BM.RAIL_FILLET, 4)]
        out["picture_rail"] = SH.moulded_obj(R, "picture_rail", BM.wall_runs(cfg), rail, [mats["trim"]])
    if "right" not in cfg["open"]:
        fr = BM.wall_frame(cfg, "right")
        zt = d["z1"]
        cas = [wbox(fr, d["y0"] - CW, d["y0"], 0.0, zt + CW, -CT, BURY),
               wbox(fr, d["y1"], d["y1"] + CW, 0.0, zt + CW, -CT, BURY),
               wbox(fr, d["y0"] - CW - 0.012, d["y1"] + CW + 0.012, zt, zt + CW, -CT - 0.004, BURY)]
        out["door_casing"] = SH.boxes_obj(R, "door_casing", [(cas, 0.003, 3)], [mats["trim"]])
    return out


# ---------------------------------------------------------------------------------------------------- door


def build_door(R, cfg, mats):
    """The closed six-panel door in its pocket of the right wall: leaf (stiles and rails stand proud of the recessed
    panels, a bead round every panel), three hinges on the front-wall side, a lever handle with rose and escutcheon
    on the other. Objects: door_leaf (door paint), door_hardware (metal)."""
    if "right" in cfg["open"]:
        return {}
    d = cfg["door"]
    fr = BM.wall_frame(cfg, "right")
    wl0, wl1 = BM.LEAF_W0, BM.LEAF_W1
    recess = wl0 + 0.012
    u0, u1 = d["y0"] + 0.004, d["y1"] - 0.004
    z0, z1 = 0.012, d["z1"] - 0.004
    sw, tr, br, mr = 0.105, 0.105, 0.19, 0.055
    boxes = [wbox(fr, u0, u1, z0, z1, recess, wl1)]            # the core, behind the panels
    beads = []
    panels = 3
    ph = (z1 - z0 - br - tr - (panels - 1) * mr) / panels
    cw = (u1 - u0 - 3 * sw) / 2
    vedges = [z0 + br + i * (ph + mr) for i in range(panels)]
    uedges = [u0 + sw, u0 + 2 * sw + cw]
    boxes += [wbox(fr, u0, u1, z0, z0 + br, wl0, recess), wbox(fr, u0, u1, z1 - tr, z1, wl0, recess)]
    boxes += [wbox(fr, u0, u1, v + ph, v + ph + mr, wl0, recess) for v in vedges[:-1]]
    boxes += [wbox(fr, u0, u0 + sw, z0, z1, wl0, recess), wbox(fr, u1 - sw, u1, z0, z1, wl0, recess),
              wbox(fr, u0 + sw + cw, u0 + 2 * sw + cw, z0, z1, wl0, recess)]
    bead, bw0 = 0.011, recess - 0.006
    for ua in uedges:                                           # a bead round the panel recess
        ub = ua + cw
        for v in vedges:
            vt = v + ph
            beads += [wbox(fr, ua, ub, v, v + bead, bw0, recess), wbox(fr, ua, ub, vt - bead, vt, bw0, recess),
                      wbox(fr, ua, ua + bead, v, vt, bw0, recess), wbox(fr, ub - bead, ub, v, vt, bw0, recess)]
    out = {"door_leaf": SH.boxes_obj(R, "door_leaf", [(boxes, 0.003, 3), (beads, 0.0015, 3)], [mats["door"]])}
    out["door_hardware"] = build_hardware(R, cfg, fr, u0, u1, z0, z1, mats)
    return out


def _turned(bm, ring, fr, origin):
    """Place a lathe made about +Z on a wall: its axis along the wall's interior normal (towards the room), its x along
    up and its y along the wall."""
    up, n = Vector((0.0, 0.0, 1.0)), Vector(fr["n"])
    u = Vector(fr["U"])
    if up.cross(u).dot(n) < 0:                                  # keep the mapping a rotation, not a mirror
        u = -u
    for rg in ring:
        for v in rg:
            co = v.co.copy()
            v.co = Vector(origin) + up * co.x + u * co.y + n * co.z


# a lever rose: a flat disc with a soft shoulder (radius, height)
ROSE = [(0.0, 0.0), (0.024, 0.0), (0.026, 0.0015), (0.026, 0.0035), (0.0245, 0.0052), (0.021, 0.006), (0.0, 0.006)]


def build_hardware(R, cfg, fr, u0, u1, z0, z1, mats):
    """Metal parts of the door: a lever handle (rose, spindle, grip) and an escutcheon on the latch side (u1), three
    hinges (flap and barrel) on the other. Flaps are bevelled, barrels and the grip have rounded ends."""
    bm = bmesh.new()
    wl0 = BM.LEAF_W0
    hu, hz = u1 - 0.07, 1.0

    def at(u, v, w):
        return Vector(BM.to_world(fr, u, v, w))
    for (cv, scale) in ((hz, 1.0), (hz - 0.20, 0.6)):           # rose of the lever and the escutcheon below it
        ring = bm_lathe(bm, [(r * scale, z * scale) for r, z in ROSE], segs=24, mat=0, smooth=True, uv=False)
        _turned(bm, ring, fr, at(hu, cv, wl0))
    bm_tube(bm, [at(hu, hz, wl0 - 0.006), at(hu, hz, wl0 - 0.034)], 0.0075, sides=12, mat=0, cap="none", smooth=True,
            uv=False)
    grip = [at(hu, hz, wl0 - 0.034), at(hu - 0.040, hz - 0.003, wl0 - 0.036), at(hu - 0.115, hz - 0.011, wl0 - 0.036)]
    bm_tube(bm, grip, 0.0085, sides=10, mat=0, cap="round", smooth=True, uv=False)
    hinges = (z0 + 0.20, (z0 + z1) / 2 + 0.03, z1 - 0.20)
    for hzv in hinges:
        bm_tube(bm, [at(u0, hzv - 0.045, wl0 + 0.004), at(u0, hzv + 0.045, wl0 + 0.004)], 0.0052, sides=10, mat=0,
                cap="round", smooth=True, uv=False)
    flaps = SH.box_bm([([wbox(fr, u0, u0 + 0.036, hzv - 0.045, hzv + 0.045, wl0 - 0.002, wl0) for hzv in hinges],
                        0.0007, 2)])
    bm_append(bm, flaps)
    flaps.free()
    return _mesh(R, "door_hardware", bm, [mats["metal"]])


# ---------------------------------------------------------------------------------------------------- fixtures


def build_fixtures(R, cfg, mats):
    """A light switch and a double socket by the door, two sockets on the window wall (plates of ivory plastic)."""
    plates, inserts = [], []
    d, w = cfg["door"], cfg["window"]

    def plate(wall, u, v, pw, ph, rocker=False, sockets=0):
        fr = BM.wall_frame(cfg, wall)
        plates.append(wbox(fr, u - pw / 2, u + pw / 2, v - ph / 2, v + ph / 2, -0.009, BM.BURY))
        if rocker:                                              # inserts stand on the plate, their backs sunk 1.5 mm in
            inserts.append(wbox(fr, u - 0.014, u + 0.014, v - 0.028, v + 0.028, -0.013, -0.0075))
        for k in range(sockets):
            su = u + (k - (sockets - 1) / 2) * 0.072
            inserts.append(wbox(fr, su - 0.024, su + 0.024, v - 0.024, v + 0.024, -0.0115, -0.0075))

    if "right" not in cfg["open"]:
        u = d["y1"] + BM.CASING_W + 0.13
        plate("right", u, 1.25, 0.085, 0.085, rocker=True)
        plate("right", u, 0.30, 0.085, 0.085, sockets=1)
    if "back" not in cfg["open"]:
        plate("back", w["x1"] + BM.CASING_W + 0.32, 0.30, 0.155, 0.085, sockets=2)
        plate("back", w["x0"] - BM.CASING_W - 0.30, 0.30, 0.085, 0.085, sockets=1)
    out = {}
    if plates:
        out["fixtures"] = SH.boxes_obj(R, "fixtures", [(plates, 0.0035, 4)], [mats["plate"]])      # a soft front face
        out["fixture_inserts"] = SH.boxes_obj(R, "fixture_inserts", [(inserts, 0.0009, 3)], [mats["insert"]])
    return out


# ---------------------------------------------------------------------------------------------------- neon, rose


def neon_line(cfg):
    """End points (set coordinates) of the neon tube and the outward-into-the-room direction of its wall."""
    fr = BM.wall_frame(cfg, cfg["neon_wall"])
    v = cfg["rail_z"] + 0.030 + 0.0085
    a = BM.to_world(fr, fr["u0"] + 0.14, v, -0.014)
    b = BM.to_world(fr, fr["u1"] - 0.14, v, -0.014)
    return Vector(a), Vector(b), Vector(fr["n"])


CLIP_RING = [(0.0080, -0.0050), (0.0097, -0.0050), (0.0105, -0.0042), (0.0105, 0.0042), (0.0097, 0.0050),
             (0.0080, 0.0050), (0.0076, 0.0046), (0.0076, -0.0046), (0.0080, -0.0050)]    # (radius, along the tube)


def _oriented(bm, rings, origin, axis, up=(0.0, 0.0, 1.0)):
    """Place a lathe made about +Z: its axis along `axis`, its x as close to `up` as the axis allows, at `origin`."""
    ez = Vector(axis).normalized()
    ex = Vector(up) - ez * Vector(up).dot(ez)
    ex = ex.normalized() if ex.length > 1e-6 else Vector((1.0, 0.0, 0.0))
    ey = ez.cross(ex)
    M = Matrix(((ex.x, ey.x, ez.x, origin[0]), (ex.y, ey.y, ez.y, origin[1]), (ex.z, ey.z, ez.z, origin[2]),
                (0.0, 0.0, 0.0, 1.0)))
    bmesh.ops.transform(bm, matrix=M, verts=[v for rg in rings for v in rg])


def build_neon(R, cfg, mats):
    """A pink neon tube lying on the picture rail of one wall (spec neon_wall), held by four ring clips on tabs."""
    if cfg["neon"] is None or cfg["neon_wall"] in cfg["open"]:
        return {}
    a, b, _ = neon_line(cfg)
    bm = bmesh.new()
    n = 24
    bm_tube(bm, [a.lerp(b, i / n) for i in range(n + 1)], 0.0075, sides=12, mat=0, cap="round", smooth=True, uv=False)
    tube = _mesh(R, "neon", bm, [mats["neon"]], smooth=True)
    fr = BM.wall_frame(cfg, cfg["neon_wall"])
    vc = cfg["rail_z"] + 0.030 + 0.0085
    clips = bmesh.new()
    length = fr["u1"] - fr["u0"] - 0.28
    for i in range(4):
        u = fr["u0"] + 0.14 + length * i / 3
        centre = Vector(BM.to_world(fr, u, vc, -0.014))
        ring = bm_lathe(clips, CLIP_RING, segs=16, mat=0, smooth=True, uv=False)
        _oriented(clips, ring, centre, fr["U"])
        bm_tube(clips, [centre, Vector(BM.to_world(fr, u, vc, BM.BURY))], 0.0045, sides=8, mat=0, cap="none",
                smooth=True, uv=False)
    return {"neon": tube, "neon_clips": _mesh(R, "neon_clips", clips, [mats["trim_dark"]], smooth=True)}


def build_pendant(R, cfg, mats):
    """Ceiling rose with a short flex and a bare (off) bulb holder: a stub, not a lamp."""
    if not cfg["pendant"] or "ceiling" in cfg["open"]:
        return {}
    H = cfg["H"]
    bm = bmesh.new()
    bm_lathe(bm, [(0.0, -0.026), (0.030, -0.026), (0.066, -0.016), (0.0735, -0.010), (0.0745, -0.005), (0.0725, -0.001),
                  (0.0685, 0.0), (0.0, 0.0)], segs=32, mat=0, smooth=True, uv=False, center=(0.0, 0.0, H))
    rose = _mesh(R, "ceiling_rose", bm, [mats["plate"]], smooth=True)
    bm = bmesh.new()
    bm_tube(bm, [Vector((0, 0, H - 0.024)), Vector((0, 0, H - 0.30))], 0.0035, sides=8, mat=0, cap="none", smooth=True,
            uv=False)
    bm_lathe(bm, [(0.0, -0.354), (0.012, -0.354), (0.018, -0.351), (0.021, -0.345), (0.0205, -0.330), (0.0185, -0.315),
                  (0.0165, -0.307), (0.012, -0.301), (0.0, -0.300)],
             segs=16, mat=0, smooth=True, uv=False, center=(0.0, 0.0, H))
    flex = _mesh(R, "ceiling_flex", bm, [mats["rocker"]], smooth=True)
    bm = bmesh.new()
    bm_lathe(bm, [(0.0, -0.462), (0.016, -0.458), (0.030, -0.436), (0.034, -0.402), (0.026, -0.372), (0.012, -0.352),
                  (0.0, -0.350)], segs=20, mat=0, smooth=True, uv=False, center=(0.0, 0.0, H))
    bulb = _mesh(R, "ceiling_bulb", bm, [mats["bulb"]], smooth=True)
    return {"ceiling_rose": rose, "ceiling_flex": flex, "ceiling_bulb": bulb}


__all__ = ["build_shell", "build_floor", "build_trim", "build_door", "build_fixtures", "build_neon", "build_pendant",
           "neon_line", "wbox"]
