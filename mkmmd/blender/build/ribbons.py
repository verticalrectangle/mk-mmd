"""ribbons: 3D paint strokes that live in the scene, so the camera moves through them (docs/design.md: Ribbons).

[[ribbon]] entries (mkmmd.core.ribbon checks them and makes their centre lines) become, each:
  <name>          an empty, the ribbon's frame: the world's, or riding a bone or a prop (`ride`, a Child Of without
                  scale); its properties `grow`, `tail` (shares of the length: it shows from tail to grow) and
                  `thick` (the radius scale) are keyed here (the draw, a retract or chase, the beat pulses) and drive
                  the geometry; `mk_ribbon` keeps the centre line in its frame with the radius that holds the surface
                  at each point (x, y, z, r per point), which the `ribbon_clear` check reads
  <name>_ribbon   the surface: a mesh of the centre line under the geometry nodes of mkmmd/blender/build/ribbon_nodes.py
                  (a crayon, a striped toothpaste or a neon marker)
  <name>_glow     a neon's glow shell (the same centre line, a round tube of additive light)
in the collection Ribbons. A ribbon `from` a prop's use point (a tube's nozzle) starts inside it and leads out along its
direction; the prop's `squeeze` (a card param) is keyed over the draw (back over a retract).

Every ribbon keeps clear of the cast's bodies (rig.json, as posed over the frames it shows, before the sim): what it
would pass through is pushed out (mkmmd.core.ribbon.push_clear) by `clear` metres more than its fattest radius, the
pushes smoothed along it; the stage reports how far it moved and the least gap it measured on every frame (a WARNING
when that is under zero: bodies on both sides closer than the ribbon is fat)."""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ...core import lanes as LN
from ...core import ribbon as RB
from ...solvers import geom
from .. import keys as K
from .. import scene as S
from ..library.sets import nightkit as NK
from . import BuildError, collection, targets
from . import ribbon_nodes as RN

STRIDE = 2                 # the push measures every second frame; the report measures every frame
BEND = 0.12                # metres along the ribbon a push is smoothed over (a Gaussian's sigma): it bends round a body


def _mat(m):
    return np.array([list(r) for r in m], float)


def _member(ctx, ref, where):
    """(member, bone or None) of a "cast:<name>[.<bone>]" string or a {cast, bone?} table; (None, None) otherwise."""
    if isinstance(ref, str) and ref.startswith("cast:"):
        name, _, bone = ref[5:].partition(".")
    elif isinstance(ref, dict) and "cast" in ref:
        name, bone = ref["cast"], ref.get("bone") or ""
    else:
        return None, None
    m = ctx.cast.get(name)
    if m is None:
        raise BuildError(f"{where}: no cast member {name!r} (cast: {', '.join(sorted(ctx.cast)) or 'none'})")
    if bone:
        try:
            bone = S.resolve_bone(m.arm, bone)
        except KeyError as e:
            raise BuildError(f"{where}: {e.args[0]}") from None
    return m, bone or None


def _point(ctx, ref, where):
    try:
        return np.array(tuple(targets.point(ctx, ref)), float)
    except BuildError as e:
        raise BuildError(f"{where}: {e}") from None


def _frame_of(ob):
    """The unscaled world frame (4x4) of an object."""
    return geom.unscaled(_mat(ob.matrix_world))


# ------------------------------------------------------------------------------------------------------ inputs
def _source(ctx, r, where):
    """The `from` use point as {prop, point, dir (or None), radius (or None)} in the world at the current frame."""
    src = r["from"]
    if src is None:
        return None
    p = ctx.props.get(src["prop"])
    if p is None:
        raise BuildError(f"{where}: from: no prop {src['prop']!r} (props: {', '.join(sorted(ctx.props)) or 'none'})")
    entry = None
    for kind, items in p.card.get("use", {}).items():
        for u in items if isinstance(items, list) else []:
            if u.get("name") == src["use"] and entry is None:
                entry = u
    if entry is None:
        have = sorted({u.get("name") for items in p.card.get("use", {}).values() if isinstance(items, list)
                       for u in items})
        raise BuildError(f"{where}: from: prop {src['prop']!r} has no use point {src['use']!r} (it has {have})")
    local = entry.get("point", entry.get("center"))
    if local is None:
        raise BuildError(f"{where}: from: {src['prop']}.{src['use']} has no point")
    M = p.root.matrix_world
    if entry.get("object") and entry["object"] in bpy.data.objects:
        M = bpy.data.objects[entry["object"]].matrix_world
        local = (0.0, 0.0, 0.0) if "point" not in entry else local
    d = entry.get("dir", entry.get("axis"))
    scale = float(np.mean(M.to_scale()))
    return {"prop": src["prop"], "point": np.array(tuple(M @ Vector(local)), float),
            "dir": None if d is None else np.array(tuple((M.to_3x3() @ Vector(d)).normalized()), float),
            "radius": None if entry.get("radius") is None else float(entry["radius"]) * scale}


def _axes(ctx, ref, where):
    """The frame a path round `ref` is laid out in: a cast member's (x her left, y her back, z up: her root's turn about
    the vertical), else the world's."""
    m, _ = _member(ctx, ref, where)
    if m is None:
        return np.eye(3)
    R = _frame_of(m.root)[:3, :3]
    y = R[:, 1] - R[2, 1] * np.array([0.0, 0.0, 1.0])
    y = y / max(np.linalg.norm(y), 1e-9) if np.linalg.norm(y) > 1e-6 else np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    return np.stack([np.cross(y, z), y, z], 1)


def _geometry(ctx, r, where, src):
    """What mkmmd.core.ribbon.centre_line needs from the scene, at the current frame."""
    p = r["path"]
    kind = p["kind"]
    geo = {}
    if src is not None:
        if src["dir"] is None and kind == "fall":
            raise BuildError(f"{where}: from {src['prop']}: the use point has no direction to fall out along")
        geo["from"] = (src["point"], src["dir"])                      # no direction: it leads out toward the path
    if kind == "orbit":
        geo["center"] = _point(ctx, p["around"], f"{where}: orbit.around")
        geo["axes"] = _axes(ctx, p["around"], where)
    elif kind == "spiral":
        m, bone = _member(ctx, p["around"], where)
        axis = _point(ctx, p["around"], f"{where}: spiral.around") if bone or m is None else \
            np.array(tuple(m.root.matrix_world.translation), float)
        floor = float(m.root.matrix_world.translation.z) if m is not None else float(axis[2])
        geo["base"] = np.array([axis[0], axis[1], floor])
        geo["axes"] = _axes(ctx, p["around"], where)
        geo["top"] = (float(targets.eyes(m).z) - floor + 0.25) if m is not None else 1.8
    elif kind == "swoop":
        m = None
        if p["cast"] is not None:
            m = ctx.cast.get(p["cast"])
            if m is None:
                raise BuildError(f"{where}: swoop.cast: no cast member {p['cast']!r}")
        pts = []
        for i, q in enumerate(p["points"]):
            if isinstance(q, list) and m is not None:
                pts.append(np.array(tuple(m.root.matrix_world @ Vector(q)), float))
            else:
                pts.append(_point(ctx, q, f"{where}: swoop.points[{i}]"))
        geo["points"] = np.array(pts)
    return geo


def _ride(ctx, r, where):
    """What the ribbon rides: None (the world), ("bone", armature, bone) or ("object", name)."""
    ride = r["ride"]
    if ride is None:
        return None
    if ride is True:
        p = r["path"]
        ref = p.get("around") if p["kind"] in ("orbit", "spiral") else (
            f"cast:{p['cast']}" if p["kind"] == "swoop" and p.get("cast") else None)
        if ref is None and r["from"] is not None:
            ref = f"prop:{r['from']['prop']}"
        if not isinstance(ref, (str, dict)) or (isinstance(ref, str) and not ref.startswith(("cast:", "prop:"))):
            raise BuildError(f"{where}: ride = true: the path goes round no cast member or prop; name what it rides "
                             f"(\"cast:<name>.<bone>\", \"prop:<name>\")")
        ride = ref
    if isinstance(ride, str) and ride.startswith("prop:"):
        p = ctx.props.get(ride[5:])
        if p is None:
            raise BuildError(f"{where}: ride: no prop {ride[5:]!r}")
        return ("object", p.root.name)
    m, bone = _member(ctx, ride, where)
    if m is None:
        raise BuildError(f"{where}: ride = {ride!r}: \"cast:<name>[.<bone>]\" or \"prop:<name>\"")
    return ("bone", m.arm.name, bone) if bone else ("object", m.root.name)


def _ride_frame(ride, dg=None):
    """The unscaled world frame (4x4) of what a ribbon rides, as evaluated now."""
    if ride[0] == "bone":
        arm = bpy.data.objects[ride[1]]
        ae = arm.evaluated_get(dg) if dg is not None else arm
        return geom.unscaled(_mat(ae.matrix_world @ ae.pose.bones[ride[2]].matrix))
    ob = bpy.data.objects[ride[1]]
    return geom.unscaled(_mat((ob.evaluated_get(dg) if dg is not None else ob).matrix_world))


def _hits(ctx, r, where):
    pu = r["pulse"]
    if pu is None:
        return []
    if pu["hits"] is not None:
        return pu["hits"]
    hits_file = (ctx.data.get("audio") or {}).get("hits")
    if not hits_file:
        raise BuildError(f"{where}: pulse.lane reads the [audio] hits file (write it with `mk timeline hits`)")
    c = pu["lane"]
    try:
        ts = LN.pick(LN.load(ctx.path(hits_file)), c["lane"], c["min_db"], c["every"], c["t0"], c["t1"])
    except LN.LaneError as e:
        raise BuildError(f"{where}: pulse: {e}") from None
    if not ts:
        raise BuildError(f"{where}: pulse: lane {c['lane']!r} has no hit there (from, to, min_db)")
    return ts


def _frames(ctx, r):
    """The Blender frames a ribbon shows on (any of it), inside the scene's range."""
    f0 = math.floor(ctx.frame(r["draw"][0]))
    end = r["retract"] or r["chase"]
    f1 = math.ceil(ctx.frame(end[1])) if end else ctx.end
    return list(range(max(f0, ctx.start), min(f1, ctx.end) + 1))


# ------------------------------------------------------------------------------------------------------ the bodies
class _Scene:
    """The cast's collision bodies (rig.json) and what the ribbons ride, sampled once on the frames they need."""

    def __init__(self, ctx, members):
        self.shapes = geom.Shapes()
        self.missing = []
        for name in members:
            m = ctx.cast[name]
            if not m.rig or not m.rig.get("bodies"):
                self.missing.append(name)
                continue
            geom.model_shapes(self.shapes, m.rig, m.arm.name)
        self.P = self.shapes.pack()
        self.sources = [tuple(s) for s in self.shapes.sources]
        owner = {m.arm.name: n for n, m in ctx.cast.items()}
        order = [self.sources[j] for kind in geom.KINDS for j in self.P[kind]["src"]]
        self.labels = [f"{owner.get(arm, arm)}:{bone}" for _, arm, bone in order]
        self.member = np.array([owner.get(arm, arm) for _, arm, _ in order], dtype=object)
        self.count = len(order)
        self.world, self.rides = {}, {}

    def mask(self, names):
        """The bodies (in `depth`'s order) of these cast members."""
        return np.isin(self.member, list(names)) if self.count else np.zeros(0, bool)

    def sample(self, frames, rides):
        """Bodies (geom.world) and ride frames on every frame of `frames`."""
        sc = bpy.context.scene
        arms = {o.name: o for o in bpy.data.objects if o.type == "ARMATURE"}
        todo = [f for f in frames if f not in self.world]
        for f in todo:
            sc.frame_set(int(f))
            dg = bpy.context.evaluated_depsgraph_get()
            evald = {}
            pos, rot = np.zeros((len(self.sources), 3)), np.tile(np.eye(3), (len(self.sources), 1, 1))
            for j, (kind, owner, bone) in enumerate(self.sources):
                if kind != "bone":
                    continue
                if owner not in evald:
                    evald[owner] = arms[owner].evaluated_get(dg)
                ae = evald[owner]
                M = geom.unscaled(_mat(ae.matrix_world @ ae.pose.bones[bone].matrix))
                pos[j], rot[j] = M[:3, 3], M[:3, :3]
            self.world[f] = geom.world(self.P, rot, pos) if self.count else {}
            for ride in rides:
                self.rides[(ride, f)] = _ride_frame(ride, dg)
        for ride in rides:
            for f in frames:
                if (ride, f) not in self.rides:
                    sc.frame_set(int(f))
                    self.rides[(ride, f)] = _ride_frame(ride, bpy.context.evaluated_depsgraph_get())

    def depth(self, X, rad, f, use=None):
        """(depth (n,): how far each point's sphere of radius `rad` sinks into the nearest body, < 0 the gap; normal (n, 3)
        the way out of it; body index (n,), -1 for none) on frame f, counting the bodies `use` (a mask) lets in."""
        res = geom.penetration(X, rad, self.world[f], None)
        n = len(X)
        best, nrm, idx = np.full(n, -np.inf), np.zeros((n, 3)), np.full(n, -1)
        base, rows = 0, np.arange(n)
        for kind in geom.KINDS:
            if kind not in res:
                continue
            pen, nn = res[kind]
            k = pen.shape[1]
            if use is not None:
                pen = np.where(use[base:base + k][None, :], pen, -np.inf)
            j = np.argmax(pen, axis=1)
            v = pen[rows, j]
            better = v > best
            best[better], nrm[better], idx[better] = v[better], nn[rows, j][better], base + j[better]
            base += k
        return best, nrm, idx


def _rel(scene, ride, f, ref):
    """The motion (4x4) of what a ribbon rides from the reference frame to frame f (identity: it rides nothing)."""
    if ride is None:
        return np.eye(4)
    return scene.rides[(ride, f)] @ np.linalg.inv(ref)


def _windows(ctx, r, frames, s):
    """(F, n) mask of the centre-line points shown on each frame."""
    L = float(s[-1])
    return np.array([RB.visible(s, L, *RB.clock(r, ctx.time(f))) for f in frames])


def _keep_clear(ctx, r, line, rad, margin, scene, bodies, ride, ref, frames):
    """Push the centre line clear of the bodies (`bodies`: a mask) on `frames`, every STRIDE-th of them: (points, the
    push's report)."""
    use = frames[::STRIDE] if len(frames) > 2 * STRIDE else frames
    shown = _windows(ctx, r, use, line["s"])
    rels = [_rel(scene, ride, f, ref) for f in use]

    def measure(P):
        need, out = np.full(len(P), -np.inf), np.zeros((len(P), 3))
        for k, f in enumerate(use):
            vis = np.flatnonzero(shown[k])
            if not len(vis):
                continue
            R, t = rels[k][:3, :3], rels[k][:3, 3]
            d, nrm, _ = scene.depth(P[vis] @ R.T + t, rad[vis] + margin, f, bodies)
            better = d > need[vis]
            need[vis[better]] = d[better]
            out[vis[better]] = nrm[better] @ R                      # back into the reference frame
        return need, out

    sigma = max(4.0, BEND / max(float(np.median(np.diff(line["s"]))), 1e-4))
    return RB.push_clear(line["P"], measure, pinned=line["pinned"], sigma=sigma)


def _least_gap(ctx, r, line, rad, scene, bodies, ride, ref, frames):
    """The least gap (m) between the ribbon as it shows (its pulses too) and the bodies over every frame: (gap, frame,
    body label)."""
    shown = _windows(ctx, r, frames, line["s"])
    best = (np.inf, None, None)
    for k, f in enumerate(frames):
        vis = np.flatnonzero(shown[k])
        if not len(vis):
            continue
        M = _rel(scene, ride, f, ref)
        thick = RB.value_at(line["thick"], ctx.time(f))
        d, _, idx = scene.depth(line["P"][vis] @ M[:3, :3].T + M[:3, 3], rad[vis] * thick, f, bodies)
        j = int(np.argmax(d))
        if -d[j] < best[0]:
            best = (float(-d[j]), int(f), scene.labels[idx[j]] if idx[j] >= 0 else None)
    return best


# ------------------------------------------------------------------------------------------------------ objects
def _colour(ctx, c, where):
    try:
        return NK.rgb(ctx.palette, c)
    except (KeyError, ValueError, AttributeError) as e:
        raise BuildError(f"{where}: colour {c!r}: {e} (palette slots: {', '.join(sorted(ctx.palette))})") from None


def _mesh(name, P, F, wscale, s, mats):
    """The centre-line mesh: a vertex per point, edges between, the point attributes mk_f, mk_w, mk_s."""
    old = bpy.data.meshes.get(name)
    if old is not None and old.users == 0:
        bpy.data.meshes.remove(old)
    me = bpy.data.meshes.new(name)
    n = len(P)
    me.vertices.add(n)
    me.vertices.foreach_set("co", np.asarray(P, np.float32).ravel())
    me.edges.add(n - 1)
    me.edges.foreach_set("vertices", np.stack([np.arange(n - 1), np.arange(1, n)], 1).astype(np.int32).ravel())
    for key, vals, kind in (("mk_f", F, "FLOAT_VECTOR"), ("mk_w", wscale, "FLOAT"), ("mk_s", s, "FLOAT")):
        a = me.attributes.new(key, kind, "POINT")
        a.data.foreach_set("vector" if kind == "FLOAT_VECTOR" else "value", np.asarray(vals, np.float32).ravel())
    for m in mats:
        me.materials.append(m)
    me.update()
    return me


def _drive(md, nt, root):
    """The modifier's Grow, Tail, Thick follow the root's grow, tail, thick (one-variable average drivers)."""
    for key, ident in RN.inputs(nt).items():
        fc = md.driver_add(f'["{ident}"]')
        d = fc.driver
        d.type = "AVERAGE"
        v = d.variables.new()
        v.name, v.type = key.lower(), "SINGLE_PROP"
        v.targets[0].id_type, v.targets[0].id, v.targets[0].data_path = "OBJECT", root, f'["{key.lower()}"]'


def _surface(name, me, coll, root, nt, shadow):
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.parent = root
    ob.visible_shadow = shadow
    md = ob.modifiers.new("Ribbon", "NODES")
    md.node_group = nt
    _drive(md, nt, root)
    return ob


def _objects(ctx, coll, r, line, w, th, ride, ref, rad1, where):
    """The root, the surface and a neon's glow; the root's properties and stored centre line."""
    name = r["name"]
    for n in (name, f"{name}_ribbon", f"{name}_glow"):
        if bpy.data.objects.get(n) is not None:
            raise BuildError(f"{where}: an object {n!r} exists already (ribbon names must be new)")
    root = bpy.data.objects.new(name, None)
    root.empty_display_size = 0.15
    coll.objects.link(root)
    inv = np.linalg.inv(ref) if ride is not None else np.eye(4)
    P = line["P"] @ inv[:3, :3].T + inv[:3, 3]
    F = line["F"] @ inv[:3, :3].T
    if ride is not None:
        con = root.constraints.new("CHILD_OF")
        con.name = "mk_ribbon_ride"
        con.target = bpy.data.objects[ride[1]]
        if ride[0] == "bone":
            con.subtarget = ride[2]
        con.use_scale_x = con.use_scale_y = con.use_scale_z = False
        con.inverse_matrix = Matrix.Identity(4)
    for key, val, doc in (("grow", 0.0, "0..1: the front, as a share of the ribbon's length"),
                          ("tail", 0.0, "0..1: where it starts showing, as a share of its length"),
                          ("thick", 1.0, "the radius scale (the beat pulses)")):
        root[key] = val
        root.id_properties_ui(key).update(description=doc, min=0.0, soft_min=0.0, soft_max=1.0 if key != "thick" else 3.0)
    root["mk_ribbon"] = np.concatenate([P, rad1[:, None]], 1).astype(float).ravel().tolist()
    st = r["style"]
    if st == "toothpaste":
        mats = [RN.paste(f"{name}_stripe{i}", _colour(ctx, c, where)) for i, c in enumerate(r["stripes"])]
    elif st == "crayon":
        mats = [RN.crayon(f"{name}_crayon", _colour(ctx, r["color"], where), w)]
    else:
        mats = [RN.neon(f"{name}_neon", _colour(ctx, r["color"], where), r["glow"])]
    spacing = min(0.01, w / 4.0)                                    # the surface's rings (denser at the domes)
    nt = RN.tree(f"mk_ribbon_{name}", st, 0.5 * w, th / w, spacing, mats)
    obs = [_surface(f"{name}_ribbon", _mesh(f"{name}_ribbon", P, F, line["wscale"], line["s"], mats), coll, root, nt,
                    r["shadow"])]
    if r["halo"] is not None:
        glow = RN.halo(f"{name}_glow", _colour(ctx, r["color"], where), r["glow"] * r["halo"]["strength"])
        gt = RN.tree(f"mk_ribbon_{name}_glow", "halo", r["halo"]["size"] * 0.5 * w, 1.0, spacing * 2.0, [glow])
        obs.append(_surface(f"{name}_glow", _mesh(f"{name}_glow", P, F, line["wscale"], line["s"], [glow]), coll, root,
                            gt, False))
    return root, obs


def _keys(ctx, r, root, line, src, where):
    """grow / tail on every frame of their windows, thick on the pulses, the source prop's squeeze. Returns the report."""
    rows = RB.clock_keys(r, ctx.fps, ctx.frame0)
    frames = [f for f, _, _ in rows]
    K.key_prop(root, '["grow"]', frames, [g for _, g, _ in rows], interp="LINEAR")
    if r["chase"] is not None:
        K.key_prop(root, '["tail"]', frames, [t for _, _, t in rows], interp="LINEAR")
    out = {"draw": [round(ctx.frame(t), 2) for t in r["draw"]]}
    for k in ("retract", "chase"):
        if r[k] is not None:
            out[k] = [round(ctx.frame(t), 2) for t in r[k]]
    if line["thick"]:
        K.key_prop(root, '["thick"]', [ctx.frame(t) for t, _ in line["thick"]], [v for _, v in line["thick"]],
                   interp="LINEAR")
        out["pulses"] = len(line["hits"])
    if src is not None and r["squeeze"] is not False:
        p = ctx.props[src["prop"]]
        has = "squeeze" in p.root.keys()
        if r["squeeze"] and not has:
            raise BuildError(f"{where}: squeeze: prop {src['prop']!r} has no `squeeze` property (its card's params)")
        if has:
            lo, hi = r["squeeze"] if isinstance(r["squeeze"], tuple) else (0.0, 1.0)
            pts = RB.squeeze_keys(r, lo, hi)
            K.key_prop(p.root, '["squeeze"]', [ctx.frame(t) for t, _ in pts], [v for _, v in pts], interp="BEZIER")
            out["squeeze"] = f"{src['prop']}.squeeze"
    return out


# ------------------------------------------------------------------------------------------------------ the stage
def _ribbon(ctx, coll, r, scene, frames):
    name = r["name"]
    where = f"ribbon {name!r}"
    sc = bpy.context.scene
    f_ref = ctx.frame(r["at"])
    sc.frame_set(int(math.floor(f_ref)), subframe=f_ref - math.floor(f_ref))
    src = _source(ctx, r, where)
    width = r["width"] or (2.0 * src["radius"] if src is not None and src["radius"] else RB.STYLE[r["style"]]["width"])
    w, th, ds = RB.sizes(r, width)
    geo = _geometry(ctx, r, where, src)
    ride = _ride(ctx, r, where)
    ref = _ride_frame(ride, bpy.context.evaluated_depsgraph_get()) if ride is not None else np.eye(4)
    try:
        line = RB.centre_line(r, geo, w, ds)
    except RB.RibbonError as e:
        raise BuildError(f"{where}: {e}") from None
    pu = r["pulse"]
    line["hits"] = _hits(ctx, r, where)
    line["thick"] = RB.pulse_keys(line["hits"], pu["gain"], pu["attack"], pu["decay"]) if pu else []
    thick_max = max([v for _, v in line["thick"]] + [1.0])
    rad1 = RB.envelope(r, w, line["wscale"], 1.0)
    rep = {"style": r["style"], "path": r["path"]["kind"], "width": round(w, 4)}
    if r["style"] != "toothpaste":
        rep["thickness"] = round(th, 4)
    if src is not None:
        rep["from"] = f"{src['prop']}.{r['from']['use']}"
    if ride is not None:
        rep["ride"] = ":".join(ride[1:])
    if r["clear"] is not None and scene.count and frames:
        bodies = scene.mask(r["clear"]["cast"] or ctx.cast)
        scene.sample(frames, [ride] if ride is not None else [])
        P, push = _keep_clear(ctx, r, line, rad1 * thick_max, r["clear"]["margin"], scene, bodies, ride, ref, frames)
        line["P"], line["F"], line["wscale"], line["s"] = RB.respace(P, line["F"], line["wscale"], ds)
        rad1 = RB.envelope(r, w, line["wscale"], 1.0)
        gap, at, body = _least_gap(ctx, r, line, rad1, scene, bodies, ride, ref, frames)
        rep["clear"] = {"margin": r["clear"]["margin"], "pushed_points": push["points"],
                        "moved_mm": round(push["moved_m"] * 1000.0, 1), "rounds": push["rounds"],
                        "least_gap_mm": round(gap * 1000.0, 1) if math.isfinite(gap) else None, "at_frame": at,
                        "body": body, "frames": len(frames)}
        if math.isfinite(gap) and gap < 0.0:
            ctx.log("WARNING", f"{where}: {-gap * 1000.0:.0f} mm into {body} at frame {at} (bodies on both sides closer "
                               f"than it is fat: move the path or make it thinner)")
    root, obs = _objects(ctx, coll, r, line, w, th, ride, ref, rad1, where)
    rep.update(_keys(ctx, r, root, line, src, where))
    rep.update(points=len(line["P"]), length_m=round(float(line["s"][-1]), 3), objects=[o.name for o in obs])
    ctx.log("ribbon", name, r["style"], r["path"]["kind"], f"{rep['length_m']} m", f"{rep['points']} points")
    return rep


def run(ctx):
    specs = ctx.data.get("ribbon") or []
    if isinstance(specs, dict):
        specs = [specs]
    if not specs:
        return {}
    if not isinstance(specs, list):
        raise BuildError("[[ribbon]]: expected an array of tables")
    try:
        checked = [RB.check(s, f"ribbon {i}" if not isinstance(s, dict) or not s.get("name") else None)
                   for i, s in enumerate(specs)]
    except RB.RibbonError as e:
        raise BuildError(str(e)) from None
    seen = set()
    for r in checked:
        if r["name"] in seen:
            raise BuildError(f"ribbon {r['name']!r}: two [[ribbon]] entries of that name")
        seen.add(r["name"])
        if r["clear"] is not None and r["clear"]["cast"]:
            missing = sorted(set(r["clear"]["cast"]) - set(ctx.cast))
            if missing:
                raise BuildError(f"ribbon {r['name']!r}: clear.cast: no cast member {missing} "
                                 f"(cast: {', '.join(sorted(ctx.cast)) or 'none'})")
    members = sorted({n for r in checked if r["clear"] is not None for n in (r["clear"]["cast"] or ctx.cast)})
    scene = _Scene(ctx, members)
    for n in scene.missing:
        ctx.log("WARNING", f"ribbons: cast member {n!r} has no collision bodies (rig.json): ribbons are not kept clear of it")
    coll = collection("Ribbons")
    sc = bpy.context.scene
    keep = sc.frame_current
    out = {}
    try:
        for r in checked:
            out[r["name"]] = _ribbon(ctx, coll, r, scene, _frames(ctx, r))
    finally:
        sc.frame_set(keep)
    return out
