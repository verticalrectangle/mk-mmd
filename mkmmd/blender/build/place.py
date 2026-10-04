"""place: the props stage's placement rules (docs/design.md: Placement), the Blender side of mkmmd.core.place.

`[[prop]] place = {on, at, facing, align, clear, avoid, distance, bearing, seed}` and `[[scatter]]` are read here; the
maths (footprints, overlap, alignment, facing, near, scatter) is mkmmd.core.place. This module only gathers what the maths
needs from the scene (surfaces from cards and their owners' matrices, footprints from bounds, prisms of what is already
placed, targets) and writes the result back as the prop root's world matrix."""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ...core import place as PL
from ...core import propcard as PC
from . import BuildError, targets

PLACE_KEYS = {"on", "at", "facing", "align", "clear", "avoid", "seed", "distance", "bearing"}
SCATTER_KEYS = {"name", "props", "on", "count", "region", "min_dist", "yaw", "seed", "clear", "avoid", "slots",
                 "card_extra"}
CLEAR = 0.02
MESH_TYPES = {"MESH", "CURVE", "FONT", "SURFACE", "META"}


def local_bounds(root):
    """(lo, hi) of the visible geometry under `root`, in the root's own frame: meshes and curves that are rendered and
    not collider shapes. A prop without geometry has zero bounds at its origin."""
    bpy.context.view_layer.update()
    inv = root.matrix_world.inverted()
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in root.children_recursive:
        if o.type not in MESH_TYPES or o.get("mk_collider") or o.hide_render:
            continue
        ob = o.evaluated_get(dg) if o.modifiers else o
        M = inv @ o.matrix_world
        pts += [np.array(M @ Vector(c)) for c in ob.bound_box]
    if not pts:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    P = np.array(pts)
    return tuple(float(x) for x in P.min(0)), tuple(float(x) for x in P.max(0))


def local_layers(root, step=0.05):
    """Height layers of the visible geometry under `root`, in its frame: [(z0, z1, x0, y0, x1, y1), ...], the plan
    bounding box of the surface points in each slice (mkmmd.core.place.layers_of_points). Objects with modifiers add the
    corners of their evaluated bounds, so generated geometry is covered too."""
    bpy.context.view_layer.update()
    inv = np.array(root.matrix_world.inverted())
    dg = bpy.context.evaluated_depsgraph_get()
    tris, pts = [], []
    for o in root.children_recursive:
        if o.type not in MESH_TYPES or o.get("mk_collider") or o.hide_render:
            continue
        M = inv @ np.array(o.matrix_world)
        if o.type == "MESH" and len(o.data.polygons):
            me = o.data
            n = len(me.vertices)
            co = np.empty(n * 3, np.float32)
            me.vertices.foreach_get("co", co)
            V = co.reshape(-1, 3).astype(float) @ M[:3, :3].T + M[:3, 3]
            me.calc_loop_triangles()
            idx = np.empty(len(me.loop_triangles) * 3, np.int32)
            me.loop_triangles.foreach_get("vertices", idx)
            if len(idx):
                tris.append(V[idx.reshape(-1, 3)])
        if o.modifiers or o.type != "MESH" or not len(o.data.polygons):
            ob = o.evaluated_get(dg)
            pts.append(np.array([M @ np.array([*c, 1.0]) for c in ob.bound_box])[:, :3])
    cloud = [PC.sample_points(np.concatenate(tris), 0.03)] if tris else []
    cloud += pts
    if not cloud:
        return []
    return [tuple(round(float(v), 5) for v in layer) for layer in PL.layers_of_points(np.concatenate(cloud), step)]


def bounds_of(prop):
    """(lo, hi) of a prop: its card's `bounds` ({min, max}) when it has them, else measured."""
    b = prop.card.get("bounds")
    if b:
        return tuple(b["min"]), tuple(b["max"])
    return local_bounds(prop.root)


def footprint_of(prop):
    lo, hi = bounds_of(prop)
    return PL.Footprint(prop.name, lo, hi, prop.card.get("front", "-Y"), bool(prop.card.get("flat", False)),
                        str(prop.card.get("origin", "")).startswith("wall"),
                        tuple(tuple(float(v) for v in layer) for layer in prop.card.get("layers", ())))


def prism_of(prop):
    """The overall prism of a prop (its bounds): what `near:` and `facing` measure against."""
    lo, hi = bounds_of(prop)
    return PL.prism_of(prop.name, lo, hi, np.array(prop.root.matrix_world), bool(prop.card.get("flat", False)))


def prisms_of(prop):
    """What a placed prop occupies, layer by layer (what blocks other props)."""
    lo, hi = bounds_of(prop)
    return PL.prisms_of(prop.name, lo, hi, np.array(prop.root.matrix_world), bool(prop.card.get("flat", False)),
                        tuple(tuple(layer) for layer in prop.card.get("layers", ())))


class Target:
    """What a `facing` or `near:` reference points at: a world plan point, plus the footprint of a prop."""

    def __init__(self, point, rect=None, yaw=0.0, name=""):
        self.point, self.rect, self.yaw, self.name = point, rect, yaw, name


class Placer:
    def __init__(self, ctx):
        self.ctx = ctx

    # ---------------------------------------------------------------- what is already there
    def others(self):
        """Prisms of every placed prop that blocks (card `blocks` false opts out) and the sets' obstacles."""
        bpy.context.view_layer.update()
        out = []
        for p in self.ctx.props.values():
            if p.card.get("blocks", True) is not False:
                out += prisms_of(p)
        for st in getattr(self.ctx, "sets", {}).values():
            for ob in st.card.get("obstacles", []):
                out.append(PL.prism_of(f"{st.name}:{ob['name']}", ob["min"], ob["max"], np.array(st.root.matrix_world)))
        return out

    def known_names(self):
        names = list(self.ctx.props)
        for st in getattr(self.ctx, "sets", {}).values():
            names += [f"{st.name}:{ob['name']}" for ob in st.card.get("obstacles", [])]
        return sorted(names)

    # ---------------------------------------------------------------- surfaces
    def surface(self, ref, who):
        """The Surface `on` names: "<prop or set>[:<use name>]", a `use.rest` plane or edge, or a `use.surface` panel."""
        if not isinstance(ref, str) or not ref:
            raise BuildError(f"{who}: place.on must be \"<prop or set>:<surface>\", got {ref!r}")
        owner_name, _, entry_name = ref.partition(":")
        owner = self.ctx.props.get(owner_name) or getattr(self.ctx, "sets", {}).get(owner_name)
        if owner is None:
            raise BuildError(f"{who}: on = {ref!r}: no prop or set {owner_name!r} (props are placed in order; have "
                             f"{sorted(self.ctx.props)}, sets {sorted(getattr(self.ctx, 'sets', {}))})")
        use = owner.card.get("use", {})
        entries = [e for e in use.get("rest", [])] + [e for e in use.get("surface", [])]
        names = [e["name"] for e in entries]
        if not entry_name:
            planes = [e for e in use.get("rest", []) if e.get("type", "plane") == "plane"]
            if len(planes) == 1:
                entry = planes[0]
            else:
                raise BuildError(f"{who}: on = {ref!r}: name a surface of {owner_name!r} ({names})")
        else:
            entry = next((e for e in entries if e["name"] == entry_name), None)
            if entry is None:
                raise BuildError(f"{who}: on = {ref!r}: {owner_name!r} has no rest plane, edge or surface "
                                 f"{entry_name!r} ({names})")
        bounds = bounds_of(owner) if owner is self.ctx.props.get(owner_name) else None      # a prop; a set has none
        try:
            return PL.surface_from_entry(f"{owner_name}:{entry['name']}", entry, np.array(owner.root.matrix_world),
                                         bounds)
        except PL.PlaceError as e:
            raise BuildError(f"{who}: on = {ref!r}: {e}")

    def host_of(self, ref):
        """The prop that owns the surface `ref` (sets own theirs through obstacles, which still block): its footprint
        never blocks what stands on it, even where its bounds rise above the surface (a headboard, a shelf)."""
        owner = ref.partition(":")[0]
        return owner if owner in self.ctx.props else ""

    # ---------------------------------------------------------------- targets
    def target(self, ref, who):
        """A Target for a prop name, a `prop:use` point, a `set:use` point, [x, y] or [x, y, z] in the world."""
        ctx = self.ctx
        if isinstance(ref, (list, tuple)) and len(ref) in (2, 3) and all(isinstance(x, (int, float)) for x in ref):
            return Target((float(ref[0]), float(ref[1])))
        if isinstance(ref, str):
            owner, _, use = ref.partition(":")
            if not use and owner in ctx.props:
                rect = prism_of(ctx.props[owner]).rect
                return Target(tuple(float(x) for x in rect.centre), rect, rect.yaw, owner)
            st = getattr(ctx, "sets", {}).get(owner)
            if st is not None:
                if not use:
                    t = st.root.matrix_world.translation
                    return Target((float(t.x), float(t.y)))
                for kind in ("look", "rest", "surface"):
                    for e in st.card.get("use", {}).get(kind, []):
                        if e["name"] == use:
                            p = st.world(e.get("point") or e.get("center") or e["a"])
                            return Target((float(p.x), float(p.y)))
                raise BuildError(f"{who}: {ref!r}: set {owner!r} has no use point {use!r}")
            if not use:
                raise BuildError(f"{who}: {ref!r} is neither a prop nor a set (props placed so far: "
                                 f"{sorted(ctx.props)})")
        try:
            p = targets.point(ctx, ref)
        except BuildError as e:
            raise BuildError(f"{who}: cannot point at {ref!r} while the props are built: {e}")
        return Target((float(p.x), float(p.y)))

    # ---------------------------------------------------------------- rules
    def rule(self, spec, who, default_yaw=0.0):
        """(surface, PL.Rule) of a `place` table."""
        bad = sorted(set(spec) - PLACE_KEYS)
        if bad:
            raise BuildError(f"{who}: unknown place key(s) {bad} (valid: {sorted(PLACE_KEYS)})")
        if "on" not in spec:
            raise BuildError(f"{who}: place needs `on` = \"<prop or set>:<surface>\"")
        surface = self.surface(spec["on"], who)
        clear = float(spec.get("clear", CLEAR))
        if clear < 0:
            raise BuildError(f"{who}: clear must be >= 0")
        avoid = spec.get("avoid", [])
        avoid = [avoid] if isinstance(avoid, str) else list(avoid)
        known = self.known_names()
        for a in avoid:
            if a not in known:
                raise BuildError(f"{who}: avoid {a!r}: no such prop (placed so far: {known})")
        facing = spec.get("facing")
        face = None
        if isinstance(facing, (int, float)) and not isinstance(facing, bool):
            face = ("yaw", float(facing))
        elif facing is not None:
            face = ("point", self.target(facing, who).point)
        align = spec.get("align", [])
        align = [align] if isinstance(align, str) else list(align)
        align = [a.split(":", 1)[1] if a.startswith("edge:") else a for a in align]
        at = spec.get("at", "center")
        if isinstance(at, str) and at.startswith("near:"):
            t = self.target(at[5:], who)
            if "distance" not in spec:
                spec = dict(spec, distance=clear)
            at = PL.Near(t.rect if t.rect is not None else t.point, float(spec["distance"]), float(spec.get("bearing", 0.0)),
                         t.name, t.yaw)
        elif isinstance(at, (list, tuple)) and len(at) == 2 and all(isinstance(p, (list, tuple)) for p in at):
            at = PL.Region(tuple(float(x) for x in at[0]), tuple(float(x) for x in at[1]))
        elif isinstance(at, (list, tuple)):
            at = tuple(float(x) for x in at)
        elif isinstance(at, (int, float)) and not isinstance(at, bool):
            at = (float(at),)
        elif at != "center":
            raise BuildError(f"{who}: at must be [u, v], \"center\", \"near:<target>\" or [[u0, v0], [u1, v1]], got {at!r}")
        if "distance" in spec and not isinstance(at, PL.Near):
            raise BuildError(f"{who}: distance belongs to at = \"near:<target>\"")
        return surface, PL.Rule(at=at, yaw=float(default_yaw), facing=face, align=tuple(align), clear=clear,
                                avoid=tuple(avoid), host=self.host_of(spec["on"]), seed=int(spec.get("seed", 0)))

    def solve(self, prop, spec, who, default_yaw=0.0):
        surface, rule = self.rule(spec, who, default_yaw)
        try:
            return PL.place(surface, footprint_of(prop), self.others(), rule, name=prop.name)
        except PL.PlaceError as e:
            raise BuildError(f"{who}: {str(e).removeprefix(prop.name + ': ')}")

    def solve_scatter(self, props, spec, who):
        """Placements for the built `props` (in order) of a `[[scatter]]` table."""
        bad = sorted(set(spec) - SCATTER_KEYS)
        if bad:
            raise BuildError(f"{who}: unknown scatter key(s) {bad} (valid: {sorted(SCATTER_KEYS)})")
        surface = self.surface(spec.get("on"), who)
        avoid = spec.get("avoid", [])
        avoid = [avoid] if isinstance(avoid, str) else list(avoid)
        known = self.known_names()
        for a in avoid:
            if a not in known:
                raise BuildError(f"{who}: avoid {a!r}: no such prop (placed so far: {known})")
        region = spec.get("region")
        if region is not None and not (len(region) == 2 and all(len(p) == 2 for p in region)):
            raise BuildError(f"{who}: region must be [[u0, v0], [u1, v1]], got {region!r}")
        yaw = spec.get("yaw", [0.0, 0.0])
        yaw = (float(yaw), float(yaw)) if isinstance(yaw, (int, float)) else (float(yaw[0]), float(yaw[1]))
        try:
            return PL.scatter(surface, [footprint_of(p) for p in props], self.others(), region=region,
                              min_dist=float(spec.get("min_dist", 0.0)), yaw=yaw, clear=float(spec.get("clear", CLEAR)),
                              avoid=avoid, seed=int(spec.get("seed", 0)), names=[p.name for p in props],
                              host=self.host_of(spec["on"]))
        except PL.PlaceError as e:
            raise BuildError(f"{who}: {str(e).removeprefix('scatter: ')}")

    # ---------------------------------------------------------------- result
    @staticmethod
    def apply(root, placement):
        """Put the root where the placement says (its world matrix: parent or not)."""
        o, yaw = placement.pose.origin, placement.pose.yaw
        root.matrix_world = Matrix.Translation(Vector(o.tolist())) @ Matrix.Rotation(yaw, 4, "Z")

    def log(self, placement):
        c = placement.clearance
        near = c.get("prop")
        p = placement.pose.origin
        self.ctx.log("place", placement.name, f"on {placement.surface.name}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) yaw "
                     f"{math.degrees(placement.pose.yaw):.1f} deg, {c['edge'] * 1000:.0f} mm from the edge" +
                     (f", nearest {near[0]} {near[1] * 1000:.0f} mm" if near else ""))
