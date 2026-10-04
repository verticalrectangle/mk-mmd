"""The highway's roadside furniture beyond rails, lamps and signs over the lanes (highway.py builds the road and calls these
with itself; the keys are in its docstring):

  studs       retroreflective road studs on the lane lines (white) and the median-side edge line (gold): glints along the road
  overpasses  a road bridge over the highway: deck, parapets and piers outside the shoulders; lamp poles near it are dropped
  signs       roadside sign boards on two posts (an exit sign): each a `use.surface` panel `<name>_panel` for [[text]]
  markers     small green mile-marker plates on posts along one shoulder
  pylons      a power line along the road: lattice towers with three cables and a red light on top that blinks
  masts       radio masts: tall triangular lattices with red lights up their height, blinking

All of it is merged meshes in the set's root frame, built in the road frame at an arc length s (`Highway.put`: local +X along
the road, +Y to its left, +Z up)."""
import math

import bpy
import numpy as np

from . import nightgeo as G
from . import nightkit as K

STUDS = {"spacing": 12.0, "size": 0.11, "strength": 2.4}
OVERPASS = {"clearance": 5.5, "width": 11.0, "deck": 1.0, "overhang": 14.0, "parapet": 1.0, "lamp": 450.0}
SIGN = {"side": "right", "size": [3.4, 1.8], "height": 2.2, "offset": None}
MARKERS = {"every": 160.0, "first": 40.0, "side": "right"}
PYLONS = {"side": "left", "offset": 60.0, "spacing": 200.0, "first": 40.0, "height": 34.0, "light": 40.0}
MAST = {"offset": -120.0, "height": 95.0, "lights": 4, "light": 60.0}
BLINK = "{hi} if fmod(frame + {ph}, {period}) < {on} else {lo}"         # a red obstruction light: on 0.4 s of 1.5 s


def _opts(defaults, given, what):
    unknown = set(given) - set(defaults) - {"s", "name"}
    if unknown:
        raise ValueError(f"highway {what}: unknown keys {sorted(unknown)} (known: {sorted(set(defaults) | {'s', 'name'})})")
    return {**defaults, **given}


def lamp_exclusion(spec):
    """Arc-length ranges where no lamp pole may stand (the overpasses' decks)."""
    out = []
    for ov in spec.get("overpasses") or []:
        o = _opts(OVERPASS, ov, "overpasses")
        out.append((float(ov["s"]) - o["width"] / 2 - 6.0, float(ov["s"]) + o["width"] / 2 + 6.0))
    return out


def studs(hw, cfg, s_all):
    """Road studs every `spacing` m on each lane line between lanes of one direction (white) and on the median-side edge
    line of each carriageway (gold)."""
    c = _opts(STUDS, cfg if isinstance(cfg, dict) else {}, "studs")
    sec = hw.sec
    lines_w, lines_g = [], []
    for cw in sec.carriageways:
        offs = sorted(ln["offset"] for ln in sec.lanes if ln["dir"] == cw["dir"])
        lines_w += [0.5 * (a + b) for a, b in zip(offs[:-1], offs[1:])]
        if sec.divided:
            lines_g.append(cw["x_in"])
    s = np.arange(6.0, hw.length - 3.0, float(c["spacing"]))
    T, Lf, U = hw.path.frame(s)
    stud = G.boxes([(c["size"], c["size"] * 0.9, 0.022)], base=True)
    for lines, key, slot in ((lines_w, "studs_white", "reflect_right"), (lines_g, "studs_gold", "reflect_left")):
        mesh = G.Mesh()
        for x in lines:
            base = hw.path.point(s) + Lf * x + U * 0.004
            mesh.add(G.instance(stud, G.frame_R(T, Lf, U), base), 0)
        if len(mesh):
            hw.obj(mesh, key, [hw.material_emit(key, slot, float(c["strength"]))], shadow=False)


def overpasses(hw, specs):
    """Road bridges across the highway at arc lengths s: a deck `clearance` m over the road, `width` m along it, reaching
    `overhang` m past each edge, parapets on both sides, square piers just outside the shoulders. At night a bridge is its
    lights: gold strips along both fascias' lower edges (the tunnels' strip light) and a warm lamp under the deck over each
    carriageway (`lamp` W, real: a car passing under is washed with it)."""
    for i, ov in enumerate(specs):
        o = _opts(OVERPASS, ov, "overpasses")
        sec, s = hw.sec, float(ov["s"])
        y0, y1 = sec.x_min - o["overhang"], sec.x_max + o["overhang"]
        yc, span = 0.5 * (y0 + y1), y1 - y0
        z0 = o["clearance"]
        eye = np.eye(3)
        boxes = [(o["width"], span, o["deck"]), (0.35, span, o["parapet"]), (0.35, span, o["parapet"])]
        at = [[0.0, yc, z0 + o["deck"] / 2], [o["width"] / 2 - 0.175, yc, z0 + o["deck"] + o["parapet"] / 2],
              [-o["width"] / 2 + 0.175, yc, z0 + o["deck"] + o["parapet"] / 2]]
        for y in (sec.x_min - 2.6, sec.x_max + 2.6):                 # piers: two pairs, outside the shoulders
            for x in (-o["width"] / 2 + 1.2, o["width"] / 2 - 1.2):
                boxes.append((1.1, 1.1, z0 + 2.0))
                at.append([x, y, (z0 - 2.0) / 2])
        name = ov.get("name", f"overpass{i + 1}")
        geo = hw.put(G.place(G.boxes(boxes), np.broadcast_to(eye, (len(boxes), 3, 3)), at), s)[0]
        mesh = G.Mesh()
        mesh.add(geo, 0, spill=hw.spill(geo.V))
        hw.obj(mesh, name, [hw.mats["concrete"]])
        strips = G.place(G.boxes([(0.16, span, 0.11)] * 2), np.broadcast_to(eye, (2, 3, 3)),
                         [[o["width"] / 2 + 0.06, yc, z0 + 0.12], [-o["width"] / 2 - 0.06, yc, z0 + 0.12]])
        lit = G.Mesh()
        lit.add(hw.put(strips, s)[0], 0)
        hw.obj(lit, f"{name}_strips", [hw.mats["tunnel_light"]], shadow=False)
        T, Lf, U = hw.path.frame(np.array([s]))
        base = hw.path.point(np.array([s]))[0]
        for k, cw in enumerate(sec.carriageways):
            offs = [ln["offset"] for ln in sec.lanes if ln["dir"] == cw["dir"]]
            ld = bpy.data.lights.new(f"{hw.name}_{name}_lamp{k}", "POINT")
            ld.energy, ld.color, ld.shadow_soft_size, ld.use_shadow = float(o["lamp"]), hw.col("lamp"), 0.5, False
            ob = bpy.data.objects.new(ld.name, ld)
            hw.coll.objects.link(ob)
            ob.parent = hw.root
            ob.location = tuple(base + Lf[0] * float(np.mean(offs)) + U[0] * (z0 - 0.3))


def signs(hw, specs):
    """Roadside sign boards on two posts, facing the traffic that comes along +T: each registers `<name>_panel` as a card
    `use.surface` (centre, normal, up, size) for [[text]]."""
    eye = np.eye(3)
    for i, sg in enumerate(specs):
        o = _opts(SIGN, sg, "signs")
        name = sg.get("name", f"sign{i + 1}")
        s = float(sg["s"])
        w, h = (float(v) for v in o["size"])
        x = float(o["offset"]) if o["offset"] is not None else (
            hw.sec.x_min - 3.2 - w / 2 if o["side"] == "right" else hw.sec.x_max + 3.2 + w / 2)
        ground = hw.ground_dz(x)
        zc = ground + float(o["height"]) + h / 2
        posts = G.place(G.boxes([(0.12, 0.12, zc + h / 2 - ground)] * 2, base=True), np.broadcast_to(eye, (2, 3, 3)),
                        [[0.05, x - w * 0.32, ground], [0.05, x + w * 0.32, ground]])
        back = G.place(G.boxes([(0.08, w, h)], skip=(1,)), [eye], [[0.0, x, zc]])
        face = G.quad(w, h, "YZ", flip=True)
        face = G.Geom(face.V + np.array([-0.045, x, zc]), face.Q, None, face.UV)
        panel = G.Mesh()
        panel.add(hw.put(posts, s)[0], 0)
        panel.add(hw.put(back, s)[0], 0)
        panel.add(hw.put(face, s)[0], 1)
        pname = f"{name}_panel"
        hw.obj(panel, pname, [hw.mats["steel"], hw.sign_material(w, h)], shadow=False)
        T, Lf, U = hw.path.frame(np.array([s]))
        Rm = G.frame_R(T, Lf, U)[0]
        centre = hw.path.point(np.array([s]))[0] + Rm @ np.array([-0.046, x, zc])
        hw.surfaces.append({"name": pname, "center": centre.tolist(), "normal": (-Rm[:, 0]).tolist(), "up": Rm[:, 2].tolist(),
                            "size": [w, h]})


def markers(hw, cfg):
    """Small green mile-marker plates on posts along a shoulder, every `every` m."""
    c = _opts(MARKERS, cfg if isinstance(cfg, dict) else {}, "markers")
    s = np.arange(float(c["first"]), hw.length - 5.0, float(c["every"]))
    s = s[~G.in_ranges(s, hw.ranges, 8.0)]
    if not len(s):
        return
    x = hw.sec.x_min - 1.9 if c["side"] == "right" else hw.sec.x_max + 1.9
    ground = hw.ground_dz(x)
    T, Lf, U = hw.path.frame(s)
    post = G.boxes([(0.06, 0.06, 1.25)], base=True)
    plate = G.place(G.boxes([(0.03, 0.26, 0.52)]), [np.eye(3)], [[-0.04, 0.0, 1.0 + 0.26]])
    base = hw.path.point(s) + Lf * x + U * ground
    steel, green = G.Mesh(), G.Mesh()
    steel.add(G.instance(post, G.frame_R(T, Lf, U), base), 0)
    green.add(G.instance(plate, G.frame_R(T, Lf, U), base), 0)
    hw.obj(steel, "marker_posts", [hw.mats["steel"]], shadow=False)
    hw.obj(green, "markers", [hw.sign_material(0.26, 0.52)], shadow=False)


def _blinking(hw, key, strength, period_s=1.5, on_s=0.4, phase=0.0):
    """A red emissive material whose strength blinks (a driver on the frame; no Python at render time)."""
    m, nb = K.new_material(f"{hw.name}_{key}")
    nb.output(nb.emission(K.rgb(hw.pal, "love"), strength))
    node = next(n for n in m.node_tree.nodes if n.type == "EMISSION")
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    fc = node.inputs["Strength"].driver_add("default_value")
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = BLINK.format(hi=round(strength, 3), lo=round(strength * 0.06, 3), ph=round(phase * fps, 2),
                                period=round(period_s * fps, 2), on=round(on_s * fps, 2))
    if not (d.is_valid and d.is_simple_expression):
        raise ValueError(f"highway: the blink driver is not a simple expression ({d.expression})")
    return m


def _lattice(width_base, width_top, height, legs=4, bays=7, bar=0.12):
    """A tapering lattice tower (legs, horizontal rings and zig-zag braces on every face) as bar end points."""
    ang = [2 * math.pi * k / legs + (math.pi / 4 if legs == 4 else math.pi / 2) for k in range(legs)]
    zs = np.linspace(0.0, height, bays + 1)
    rad = lambda z: (width_base + (width_top - width_base) * z / height) / (2 * math.cos(math.pi / legs))   # noqa: E731
    pt = lambda k, z: (rad(z) * math.cos(ang[k % legs]), rad(z) * math.sin(ang[k % legs]), z)             # noqa: E731
    p0, p1 = [], []
    for k in range(legs):
        for b in range(bays):
            p0.append(pt(k, zs[b]))
            p1.append(pt(k, zs[b + 1]))                                # legs
            p0.append(pt(k, zs[b + 1]))
            p1.append(pt(k + 1, zs[b + 1]))                            # rings
            p0.append(pt(k, zs[b] if b % 2 == 0 else zs[b + 1]))
            p1.append(pt(k + 1, zs[b + 1] if b % 2 == 0 else zs[b]))   # braces
    return G.bars(p0, p1, bar, bar)


def pylons(hw, cfg):
    """A power line along the road at lateral `offset` (m, positive to the left): lattice towers every `spacing` m with two
    cross arms, three sagging cables, and a blinking red light on each top."""
    c = _opts(PYLONS, cfg if isinstance(cfg, dict) else {}, "pylons")
    off = float(c["offset"]) * (1.0 if c["side"] == "left" else -1.0)
    s = np.arange(float(c["first"]), hw.length, float(c["spacing"]))
    if len(s) < 2:
        return
    H = float(c["height"])
    tower = _lattice(5.0, 1.0, H, legs=4, bays=8, bar=0.14)
    arms = G.bars([(0.0, -7.0, 0.80 * H), (0.0, -5.0, 0.93 * H)], [(0.0, 7.0, 0.80 * H), (0.0, 5.0, 0.93 * H)], 0.35, 0.5)
    steel, cable, red = G.Mesh(), G.Mesh(), G.Mesh()
    T, Lf, U = hw.path.frame(s)
    base = hw.path.point(s) + Lf * off + U * (hw.ground_dz(off) - 0.5)
    R = G.frame_R(T, Lf, U)
    steel.add(G.instance(tower, R, base), 0)
    steel.add(G.instance(arms, R, base), 0)
    red.add(G.instance(G.boxes([(0.5, 0.5, 0.5)]), R, base + U * (H + 0.4)), 0)
    tips = [(-7.0, 0.80 * H), (7.0, 0.80 * H), (0.0, 0.93 * H + 2.5)]
    for k in range(len(s) - 1):                                      # catenaries between neighbouring towers
        for y, z in tips:
            a = base[k] + R[k] @ np.array([0.0, y, z])
            b = base[k + 1] + R[k + 1] @ np.array([0.0, y, z])
            u = np.linspace(0.0, 1.0, 13)
            pts = a[None] * (1 - u)[:, None] + b[None] * u[:, None]
            pts[:, 2] -= 0.035 * np.linalg.norm(b - a) * 4 * u * (1 - u)
            cable.add(G.bars(pts[:-1], pts[1:], 0.05, 0.05), 0)
    hw.obj(steel, "pylons", [hw.mats["steel"]], shadow=False)
    hw.obj(cable, "pylon_cables", [hw.mats["steel"]], shadow=False)
    hw.obj(red, "pylon_lights", [_blinking(hw, "pylon_red", float(c["light"]))], shadow=False)


def masts(hw, specs):
    """Radio masts at (s, offset): a slim triangular lattice `height` m tall with `lights` red lights up it, blinking."""
    for i, mt in enumerate(specs):
        o = _opts(MAST, mt, "masts")
        H = float(o["height"])
        g = _lattice(2.2, 0.9, H, legs=3, bays=max(6, int(H / 6)), bar=0.12)
        steel, red = G.Mesh(), G.Mesh()
        x = float(o["offset"])
        steel.add(hw.put(g, float(mt["s"]), lat=x, dz=hw.ground_dz(x) - 0.5)[0], 0)
        n = max(1, int(o["lights"]))
        lamp = G.place(G.boxes([(0.6, 0.6, 0.6)] * n), np.broadcast_to(np.eye(3), (n, 3, 3)),
                       [[0.0, 0.0, H * (k + 1) / n + 0.3] for k in range(n)])
        red.add(hw.put(lamp, float(mt["s"]), lat=x, dz=hw.ground_dz(x) - 0.5)[0], 0)
        name = mt.get("name", f"mast{i + 1}")
        hw.obj(steel, name, [hw.mats["steel"]], shadow=False)
        hw.obj(red, f"{name}_lights", [_blinking(hw, f"{name}_red", float(o["light"]), phase=0.37 * (i + 1))], shadow=False)
