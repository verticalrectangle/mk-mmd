"""flight: props flown relative to a target by `[[flight]]` tables (docs/design.md: Flights; the maths is
mkmmd/core/flight.py). Two stages share this module:

`flight` (after motion, before perform) hangs each flown prop on its target: an empty `<prop>_pivot` rides the target
(bone-parented to a cast member's bone at the target's point, parented to her root or a prop, or standing at a world
point), and the prop's root copies its place (a Copy Location with offset, `mk_flight`) and is keyed the flight's moves
as offsets from it, in the target's frame. Whatever moves the target later (perform, a marionette, stepped) carries the
prop with it, so a gaze at the prop in perform and a camera aimed at it find it where it flies.

`wire` (after everything that moves the target: `finish`) bakes it: the pivot's place on every frame, the path followed
`lag` late (a perch rides the target, the prop's `perch` anchor on it), the heading and bank, keyed on the root's
location and rotation (every frame, or held on the prop's step grid as the paper theatre holds its own), the wings'
`rest` with it; and for a prop whose card has a `wire` (the dragonfly) the wire from the hand to the belly on every
frame (each point of its poly curve keyed in the world), with the hand's empty. The report has per flight the legs, the
least clearance of the prop and of the wire to the target, and how far the wire's middle lags."""
import bpy
import numpy as np
from mathutils import Matrix, Vector

from ...core import flight as FL
from ...core import paper_theatre as PT
from .. import scene as S
from .. import scope as SCOPE
from ..keys import set_fcurve
from ..library.props import dragonfly as DFB
from . import BuildError, targets

CONSTRAINT = "mk_flight"


def flights(ctx):
    """[(F, prop)] of the project's [[flight]] tables, checked against the props."""
    entries = ctx.data.get("flight", [])
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        raise BuildError("[[flight]]: expected an array of tables")
    out, seen = [], set()
    for i, spec in enumerate(entries):
        try:
            F = FL.normalize(spec, i)
        except FL.FlightError as e:
            raise BuildError(str(e)) from None
        prop = ctx.props.get(F["prop"])
        if prop is None:
            raise BuildError(f"{F['where']}: no prop {F['prop']!r} (props: {', '.join(sorted(ctx.props)) or 'none'})")
        if prop.root.parent is not None:
            raise BuildError(f"{F['where']}: the prop's root has a parent ({prop.root.parent.name}): a flight places it in "
                             f"the world, so drop its `parent` / `attach`")
        if F["prop"] in seen:
            raise BuildError(f"{F['where']}: prop {F['prop']!r} is flown by two flights: one at a time")
        seen.add(F["prop"])
        out.append((F, prop))
    return out


def _cast_of(target):
    if isinstance(target, str) and target.startswith("cast:"):
        return target[5:].partition(".")[0]
    if isinstance(target, dict) and "cast" in target:
        return target["cast"]
    return None


def frame_of(ctx, target):
    """The target's frame (3, 3): a cast member's root turn for a cast target, the world's axes otherwise."""
    name = _cast_of(target)
    if name is not None and name in ctx.cast:
        return np.array(ctx.cast[name].root.matrix_world.to_3x3().normalized())
    return np.eye(3)


def _home(prop):
    return [float(v) for v in prop.root.matrix_world.translation]


def pivot(ctx, F, prop):
    """The empty `<prop>_pivot` riding the flight's target (its place set on the scene's first frame)."""
    name = f"{prop.name}_pivot"
    old = bpy.data.objects.get(name)
    if old is not None:
        bpy.data.objects.remove(old)
    pv = bpy.data.objects.new(name, None)
    pv.empty_display_size = 0.05
    for c in prop.root.users_collection:
        c.objects.link(pv)
    target = F["target"] if F["target"] is not None else _home(prop)
    sc = bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(ctx.start)
    try:
        p = targets.point(ctx, target)
    except BuildError as e:
        raise BuildError(f"{F['where']}.target: {e}") from None
    cast = _cast_of(target)
    if cast is not None:
        m = ctx.cast.get(cast)
        if m is None:
            raise BuildError(f"{F['where']}.target: no cast member {cast!r}")
        bone = (target.get("bone") if isinstance(target, dict) else target[5:].partition(".")[2]) or None
        if isinstance(target, str) and not bone:
            bone = "head"                                   # "cast:rin": her eyes, which ride her head
        if bone:
            pv.parent, pv.parent_type, pv.parent_bone = m.arm, "BONE", S.resolve_bone(m.arm, bone)
        else:
            pv.parent = m.root
    elif isinstance(target, dict) and "prop" in target:
        pv.parent = ctx.props[target["prop"]].root
    elif isinstance(target, str):
        rides = [o for o in targets.objects(ctx, target) if o is not ctx.props.get(target.partition(":")[0], prop).root]
        pv.parent = rides[0] if rides else ctx.props[target.partition(":")[0]].root
    bpy.context.view_layer.update()
    pv.matrix_world = Matrix.Translation(p)
    sc.frame_set(keep)
    return pv


def _carry(prop, pv):
    root = prop.root
    for c in list(root.constraints):
        if c.name == CONSTRAINT:
            root.constraints.remove(c)
    c = root.constraints.new("COPY_LOCATION")
    c.name, c.target, c.use_offset = CONSTRAINT, pv, True
    return c


def _eulers(M):
    """XYZ Eulers (n, 3) of rotation matrices (n, 3, 3), each continuous with the one before."""
    out, prev = [], None
    for m in M:
        e = Matrix(m.tolist()).to_euler("XYZ", prev) if prev is not None else Matrix(m.tolist()).to_euler("XYZ")
        out.append(tuple(e))
        prev = e
    return np.array(out)


def _key_root(root, frames, loc, eul, interp):
    root.rotation_mode = "XYZ"
    for c in range(3):
        set_fcurve(root, "location", c, frames, loc[:, c], interp=interp)
        set_fcurve(root, "rotation_euler", c, frames, eul[:, c], interp=interp)


def run(ctx):
    """The `flight` stage: every flown prop hung on its target and keyed its moves (see the module)."""
    todo = flights(ctx)
    ctx.flights = {}
    if not todo:
        return {}
    frames = ctx.frames.astype(float)
    ts = np.array([ctx.time(f) for f in frames])
    out = {}
    for F, prop in todo:
        home = prop.root.matrix_world.copy()                 # where it was built: its wire's hand until keyed
        pv = pivot(ctx, F, prop)
        R = frame_of(ctx, F["target"])
        pr = FL.path(F, ts)
        off = pr["O"] @ R.T
        _carry(prop, pv)
        M = FL.attitude(off, np.zeros_like(off), ts, R, pr["perch"], F["show"])
        _key_root(prop.root, frames, off, _eulers(M), "LINEAR")
        ctx.flights[prop.name] = {"pivot": pv.name, "R": R, "home": home}
        out[prop.name] = {"target": F["target"] if F["target"] is not None else "its own place", "pivot": pv.name,
                          "legs": pr["legs"]}
        ctx.log("flight", prop.name, f"{len(pr['legs'])} legs")
    return out


def _grid(ctx, card):
    """(key frames, the index of the frame each shows, interpolation) of a flown prop: its card's step grid
    (`dragonfly.step`, `step_from`) or every frame."""
    d = card.get("dragonfly", {})
    step = float(d.get("step", 0.0) or 0.0)
    if step > ctx.fps:
        raise BuildError(f"prop step = {step:g} poses a second is more than the scene's {ctx.fps:g} frames")
    keys, times, interp = PT.key_times(ctx.start, ctx.end, ctx.frame0, ctx.fps, step, float(d.get("step_from", 0.0)))
    shown = np.round(ctx.frame0 + times * ctx.fps).astype(int) - ctx.start
    return keys, np.clip(shown, 0, ctx.end - ctx.start), interp


def _sample(ctx, objs, frames, read):
    sc = bpy.context.scene
    keep = sc.frame_current
    bpy.context.view_layer.update()
    rows = []
    with SCOPE.only(objs):
        for f in frames:
            sc.frame_set(int(f))
            rows.append(read())
    sc.frame_set(keep)
    return np.array(rows, float)


def _anchor(card, name):
    for u in card.get("use", {}).get("anchor", []):
        if u.get("name") == name and "point" in u:
            return np.array(u["point"], float)
    return np.zeros(3)


def _key_wire(ob, frames, W, interp):
    ob.parent = None
    ob.matrix_world = Matrix.Identity(4)
    cu = ob.data
    pts = cu.splines[0].points
    if len(pts) != W.shape[1]:
        raise BuildError(f"{ob.name}: the wire has {len(pts)} points, the flight drew {W.shape[1]}")
    for q in range(W.shape[1]):
        for c in range(3):
            set_fcurve(cu, f"splines[0].points[{q}].co", c, frames, W[:, q, c], interp=interp)


def finish(ctx):
    """The `wire` stage: every flown prop baked where its target ended up, and its wire drawn (see the module)."""
    todo = flights(ctx)
    if not todo:
        return {}
    done = getattr(ctx, "flights", {})
    frames = ctx.frames.astype(float)
    ts = np.array([ctx.time(f) for f in frames])
    out = {}
    for F, prop in todo:
        rec = done.get(prop.name)
        pv = bpy.data.objects.get(rec["pivot"]) if rec else None
        pv = pv if pv is not None else pivot(ctx, F, prop)
        R = rec["R"] if rec else frame_of(ctx, F["target"])
        root = prop.root
        for c in list(root.constraints):
            if c.name == CONSTRAINT:
                root.constraints.remove(c)
        avoid = []
        for a in F["avoid"]:
            try:
                objs = targets.objects(ctx, a["at"])
                avoid.append((a, objs))
            except BuildError as e:
                raise BuildError(f"{F['where']}.avoid: {e}") from None
        eye_obj = None
        if F["face"] and isinstance(F["face"]["from"], str):
            eye_obj = bpy.data.objects.get(F["face"]["from"])
            if eye_obj is None:
                raise BuildError(f"{F['where']}.face.from: no object {F['face']['from']!r} (a shot camera's object: "
                                 f"`mk q BLEND --list objects`)")
        objs = [pv] + [o for _a, os in avoid for o in os] + ([eye_obj] if eye_obj else [])
        cols = _sample(ctx, objs, frames, lambda: [tuple(pv.matrix_world.translation)] +
                       [tuple(targets.point(ctx, a["at"])) for a, _o in avoid] +
                       ([tuple(eye_obj.matrix_world.translation)] if eye_obj else []))
        T = cols[:, 0]
        eye = None
        if F["face"]:
            eye = cols[:, -1] if eye_obj else np.tile(np.array(F["face"]["from"], float), (len(frames), 1))
        pr = FL.path(F, ts)
        P = FL.place(F, ts, T, R, pr, eye=eye)
        M = FL.attitude(P, T, ts, R, pr["perch"], F["show"])
        perch = _anchor(prop.card, "perch")
        P = P - pr["perch"][:, None] * np.einsum("nij,j->ni", M, perch)       # perched: its feet on the target
        keys, shown, interp = _grid(ctx, prop.card)
        _key_root(root, keys, P[shown], _eulers(M[shown]), interp)
        if "rest" in root.keys():
            set_fcurve(root, '["rest"]', 0, keys, pr["rest"][shown], interp=interp)
        d = prop.card.get("dragonfly", {})
        if d.get("objects", {}).get("variants"):                      # the wings' film stops shifting perched
            var = d["objects"]["variants"]
            objs_v = [[bpy.data.objects[names[v]] for names in var.values()] for v in range(len(next(iter(var.values()))))]
            DFB.key_films(objs_v, ctx.start, ctx.end, ctx.frame0, ctx.fps, float(d.get("step", 0.0)),
                          float(d.get("step_from", 0.0)), hold=pr["rest"] > 0.5)
        free = pr["perch"] < 0.5
        gap = np.linalg.norm(P - T, axis=1) - F["clear"]
        rep = {"frames": [int(frames[0]), int(frames[-1])], "legs": pr["legs"],
               "clear_m": round(float(gap[free].min()), 4) if free.any() else None,
               "perched_frames": int((~free).sum())}
        wire = prop.card.get("wire")
        if wire:
            belly = np.array(wire["belly"], float)
            D = P + np.einsum("nij,j->ni", M, belly)
            down = -M[:, :, 2]
            home = rec["home"] if rec else root.matrix_world
            rest_hand = np.array(home @ Vector(wire["rest_hand"]))
            H = FL.hand_track(F, ts, rest_hand)
            spheres = [(T, F["clear"])] + [(cols[:, 1 + k], a["radius"]) for k, (a, _o) in enumerate(avoid)]
            capsules = [(eye, T, 0.6 * F["face"]["margin"])] if eye is not None else []
            W = FL.wire(H, D, down, ts, spheres, F["wire"]["freq"], F["wire"]["damp"], int(wire.get("points", FL.POINTS)),
                        capsules=capsules)
            rest = FL.wire_rest(H, D, down, ts, W.shape[1])
            _key_wire(bpy.data.objects[wire["object"]], keys, W[shown], interp)
            h = bpy.data.objects[wire["hand"]]
            h.parent = None
            for c in range(3):
                set_fcurve(h, "location", c, keys, H[shown][:, c], interp=interp)
            least, at = FL.clearance(W, spheres)
            rep["wire"] = {"clear_m": round(least, 4), "at_frame": int(frames[at]) if at >= 0 else None,
                           "lag_m": round(float(np.linalg.norm(W - rest, axis=2).max()), 4),
                           "length_m": [round(float(v), 3) for v in
                                        (np.linalg.norm(np.diff(W, axis=1), axis=2).sum(1)[[0, -1]])]}
        out[prop.name] = rep
        ctx.log("flight", prop.name, f"clear {rep['clear_m']} m" + (f", wire clear {rep['wire']['clear_m']} m, lag "
                                                                     f"{rep['wire']['lag_m']} m" if wire else ""))
    return out
