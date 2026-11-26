"""marionette, set free: the build's `marionette` stage for a marionette set free and for tails (docs/design.md:
Marionette, set free; tails). mkmmd.core.marionette_free plans her (perform keys it through build/marionette.py's
`prepare`); this draws what follows from where she ended up on every frame, after `stepped`:

- the threads while she sits on them: from their anchors to where they meet her, gathered by her fist (anchor, the
  scissors' cut point, her fist, her body), each blended from its own line into the gathered one as her fist sweeps it in
  (the other hand's once that hand is up with the scissors: until then it runs straight from the control to it); the
  cut point is over her fist until the open blades come to it (BLADES_AT), then between them;
- the scissors (`scissors.prop`): lying where they lie, taken into her hand (keyed from her hand's bones on every frame:
  their grip where her fingers are, their blades along her fingers), their blades (`open`, the prop's root property)
  opening as she brings them up and shut ON the snip, let go and falling;
- after the snip: the pieces left on the falling control and the tails left on her, both ropes (mkmmd.core.
  marionette_tails), the tails held in her fist until she lets go; over a haul, her tails gathered to the hand's point
  (`<prop>_grab`), the head's and the back's run down behind her, then let go as she falls;
- for a member with no control (`tails` alone): six tails tied to her, settled on the first frame of each of her
  `shows` spans.

The ropes run on her motion as it was before it was stepped (each held pose's span filled in from it to the next) and
are drawn held on her steps, so they move with her on twos without the jolts of her steps. Her body's collision shapes
(rig.json, not her hair's, ears', skirt's or hands') keep them off her."""
import json
import math

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from ...core import marionette as MR
from ...core import marionette_free as MF
from ...core import marionette_tails as MT
from ...solvers import geom
from .. import keys as K
from .. import scope as SCOPE
from .. import scene as S
from . import BuildError
from . import marionette as MAR

HAND_BONES = ("wrist", "index1", "middle1", "little1")
FIST_IN = 0.022             # m: the bundle runs through her fist this far into the palm from her middle finger's root
GRIP = (0.06, 0.03)         # m from her wrist along her hand and out from her palm: where she holds the scissors' loops
TAIL_N = 40                 # a rope's particles (drawn with the card's points)
SKULL = 0.105               # m behind her head's middle a thread runs round the back of it (her hair's thickness)
BUNDLE_UP = 0.65            # m: the gathered threads run up together this far from the cut before they part (one cord
#                             up out of the top of a cover 1.3 m from her face)
PART_GAP = 0.015            # m each side of the cut the parted ends spring apart on the snip
BLADES_AT = (0.06, 0.15)    # m from the cut point over her fist: the open blades hold the bundle within the first, the
#                             fist alone beyond the second (on their way up from the table they never pull it aside)


def _shown(ctx, m, frames):
    """(n,) bool: the frames she is drawn in (`[[cast]] shows`, a shot's way: from the frame of `from` up to the frame of
    `to`), every frame without it."""
    spans = m.spec.get("shows")
    if not spans:
        return np.ones(len(frames), bool)
    out = np.zeros(len(frames), bool)
    for a, b in spans:
        out |= (frames >= int(round(ctx.frame(float(a))))) & (frames < int(round(ctx.frame(float(b)))))
    return out


def _held_from(frames, windows):
    """(n,) the index of the frame each frame's picture is held from: its step's inside a stepped window, its own
    outside."""
    idx = np.arange(len(frames))
    f2i = {int(f): i for i, f in enumerate(frames)}
    for w in windows:
        steps = list(w["steps"]) + [int(w["to"])]
        for a, b in zip(steps[:-1], steps[1:]):
            if a not in f2i:
                continue
            for f in range(int(a), int(b)):
                if f in f2i:
                    idx[f2i[f]] = f2i[a]
    return idx


def _unstepped(X, idx):
    """X (n, ...) sampled on frames held on steps (`idx`), each held span filled in linearly from its step to the next:
    her motion as it was before it was stepped, for the ropes to run on."""
    X = np.asarray(X, float)
    out = X.copy()
    anchors = np.nonzero(idx == np.arange(len(idx)))[0]
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b - a > 1:
            w = ((np.arange(a, b) - a) / (b - a)).reshape((-1,) + (1,) * (X.ndim - 1))
            out[a:b] = X[a] * (1.0 - w) + X[b] * w
    return out


def _orthonormal(R):
    """(n, 3, 3) the rotations nearest R (a blend of rotations)."""
    U, _s, Vt = np.linalg.svd(R)
    D = np.ones((len(R), 3))
    D[:, 2] = np.sign(np.linalg.det(U @ Vt))
    return np.einsum("nij,nj,njk->nik", U, D, Vt)


def _body_shapes(m):
    """geom.Shapes of her collision bodies (rig.json) the threads keep off: not her hair's, ears', skirt's (they move)
    nor her hands' (the tails run through her fist)."""
    rig = m.rig or {}
    smap = S.semantic_map(m.arm)
    skip = {b for c in rig.get("chains", []) for b in c["bones"]}
    for s in ("L", "R"):
        b = m.arm.data.bones.get(smap.get(f"wrist.{s}", ""))
        if b is not None:
            skip |= {b.name} | {c.name for c in b.children_recursive}
    bodies = [b for b in rig.get("bodies", []) if b.get("bone") and b["bone"] not in skip and "geom" in b
              and b["bone"] in m.arm.data.bones]
    shapes = geom.Shapes()
    geom.model_shapes(shapes, {"bodies": bodies}, m.arm.name)
    return shapes


def _sample(ctx, m, prop, card, frames, sources, control):
    """On every frame (after stepped): the anchors (A) and ends (B) of the six threads, her hands' bones (`hand`
    {(bone, side): (n, 3)}), her head, neck, chest and hips with her axes (for the threads that run round her), the
    hand's grab point and her collision shapes' sources (positions and rotations)."""
    arm = m.arm
    smap = S.semantic_map(arm)
    n = len(frames)
    anchors = [bpy.data.objects[card["anchors"][p]["object"]] for p in MR.PARTS] if control else []
    ends = [bpy.data.objects[f"{prop.name}_end_{p}"] for p in MR.PARTS]
    bars = [bpy.data.objects[v] for v in card["bars"].values()]
    grab = bpy.data.objects.get(f"{prop.name}_grab")
    hb = {(k, s): smap.get(f"{k}.{s}") for k in HAND_BONES for s in ("L", "R")}
    missing = sorted({f"{k}.{s}" for (k, s), b in hb.items() if not b})
    if missing:
        raise BuildError(f"perform.{m.name}.marionette: the model has no {', '.join(missing)} bone (her hands hold "
                         f"the threads and the scissors)")
    body_b = {k: smap[k] for k in ("head", "neck", "upper_body2", "center") if k in smap}
    out = {"A": np.zeros((n, 6, 3)), "B": np.empty((n, 6, 3)), "hand": {k: np.empty((n, 3)) for k in hb},
           "body": {k: np.empty((n, 3)) for k in body_b}, "axes": np.empty((n, 3, 3)), "grab": np.zeros((n, 3)),
           "src_pos": np.zeros((n, len(sources), 3)), "src_rot": np.tile(np.eye(3), (n, len(sources), 1, 1))}
    sc = bpy.context.scene
    keep = sc.frame_current
    bpy.context.view_layer.update()
    with SCOPE.only(anchors + ends + bars + [arm, m.root, prop.root] + ([grab] if grab else [])):
        for i, f in enumerate(frames):
            sc.frame_set(int(f))
            dg = bpy.context.evaluated_depsgraph_get()
            ae = arm.evaluated_get(dg)
            Mw = ae.matrix_world
            for j in range(6):
                if control:
                    out["A"][i, j] = anchors[j].evaluated_get(dg).matrix_world.translation
                out["B"][i, j] = ends[j].evaluated_get(dg).matrix_world.translation
            for key, b in hb.items():
                out["hand"][key][i] = Mw @ ae.pose.bones[b].head
            for key, b in body_b.items():
                out["body"][key][i] = Mw @ ae.pose.bones[b].head
            out["axes"][i] = np.array(Mw.to_3x3().normalized())
            if grab is not None:
                out["grab"][i] = grab.evaluated_get(dg).matrix_world.translation
            for j, (_kind, _owner, name) in enumerate(sources):
                M = geom.unscaled(np.array(Mw @ ae.pose.bones[name].matrix))
                out["src_pos"][i, j], out["src_rot"][i, j] = M[:3, 3], M[:3, :3]
    sc.frame_set(keep)
    return out


def _palm_sign(m, side):
    """+1 or -1: the sign that makes (index root - wrist) x (little root - wrist) point into the palm (down at rest)."""
    arm, smap = m.arm, S.semantic_map(m.arm)
    h = {k: arm.data.bones[smap[f"{k}.{side}"]].head_local for k in ("wrist", "index1", "little1")}
    nrm = (h["index1"] - h["wrist"]).cross(h["little1"] - h["wrist"])
    return 1.0 if nrm.z < 0.0 else -1.0


def _hand_frame(D, side, sign):
    """(wrist (n, 3), along her hand (n, 3), into her palm (n, 3)) from her hand's bones."""
    W, I, Mi, Li = (D["hand"][(k, side)] for k in HAND_BONES)
    along = MT._unit(Mi - W)
    palm = sign * MT._unit(np.cross(I - W, Li - W))
    palm = MT._unit(palm - np.sum(palm * along, axis=1, keepdims=True) * along)
    return W, along, palm


def _air(ctx, name, P, ts):
    """The tails' air (mkmmd.core.marionette_tails.Air): `tails.wind` ([x, y, z], or a wind table as `[sim.<cast>]`'s),
    else the member's own `[sim.<cast>] wind` (keys, gusts; its carrier left out), else still air."""
    tw = (P.get("tails") or {}).get("wind")
    spec = tw if tw is not None else ctx.section("sim", name).get("wind")
    seed = (P.get("tails") or {}).get("seed", 0)
    if spec is None:
        return MT.Air(seed=seed)
    if isinstance(spec, (list, tuple)):
        return MT.Air(np.asarray(spec, float), seed=seed)
    from ...solvers import strands
    w = {k: v for k, v in spec.items() if k != "carrier"}
    for key in ("direction", "speed"):                            # keys on clip seconds, as the sim stage passes them
        if isinstance(w.get(key), list) and w[key] and isinstance(w[key][0], (list, tuple)):
            w[key] = [[float(k[0]) - float(ts[0])] + [float(v) for v in k[1:]] for k in w[key]]
    try:
        mean = strands.Wind(w, fps=ctx.fps).mean_at(np.asarray(ts, float) - float(ts[0]))
    except ValueError as e:
        raise BuildError(f"perform.{name}.marionette.tails: wind: {e}") from None
    return MT.Air(mean, ts, seed=seed)


def _gravity(ctx, name, P, ts):
    """(n,) the tails' gravity: `tails.gravity` (a scale or keys), else the member's `[sim.<cast>] gravity` keys, else 1."""
    g = (P.get("tails") or {}).get("gravity")
    if g is None:
        g = ctx.section("sim", name).get("gravity")
    if g is None:
        return np.ones(len(ts))
    if isinstance(g, (int, float)):
        return np.full(len(ts), float(g))
    pts = sorted((float(t), float(v)) for t, v in g)
    return np.interp(ts, [t for t, _ in pts], [v for _, v in pts])


def _rope(ts, sl, idx, roots, lengths, init, pins, shapes, gravity, air, drag, settle=0.0, v_init=None):
    """(n_sl, K, TAIL_N, 3) the ropes over the frames `sl` (a slice), run on the un-stepped inputs and held on her
    steps (a step before the slice holds from the slice's first frame)."""
    out = MT.simulate(ts[sl], roots[sl], lengths, init=init, gravity=gravity[sl], air=air,
                      pins=[(k, s, P[sl], on[sl]) for k, s, P, on in pins], shapes=lambda i: shapes(sl.start + i),
                      n=TAIL_N, drag=drag, settle=settle, v_init=v_init)
    j = np.clip(idx[sl] - sl.start, 0, len(out) - 1)
    return out[j]


def _draw(prop, part, kind, frames, pts, shown, npts):
    """Key a thread's curve (`kind` thread or cut) on every frame from `pts` (n, m, 3) world points (resampled to the
    curve's points) and its visibility."""
    ob = bpy.data.objects[f"{prop.name}_{kind}_{part}"]
    Ri = np.array(prop.root.matrix_world.inverted())
    P = np.array([MT.resample(p, npts) for p in pts])
    local = P @ Ri[:3, :3].T + Ri[:3, 3]
    for q in range(npts):
        for c in range(3):
            K.set_fcurve(ob.data, f"splines[0].points[{q}].co", c, frames, local[:, q, c], interp="LINEAR")
    MAR._key_hidden(ob, frames, ~shown)


def run(ctx, name, m, P, prop, card, held_curves):
    """The stage for a marionette set free (`world.free`) or wearing tails alone (no control). Returns the report."""
    sc = bpy.context.scene
    f1 = MAR._last(ctx, m)
    frames = np.arange(ctx.start, f1 + 1)
    ts = np.array([ctx.time(f) for f in frames])
    idx = _held_from(frames, MAR._windows(ctx, name))
    shown = _shown(ctx, m, frames)
    npts = int(card["thread"]["points"])
    if not P["control"] and bpy.data.objects[f"{prop.name}_end_head"].parent is not m.arm:
        keep = sc.frame_current                                   # perform kept her body to her motion (body = false):
        sc.frame_set(int(ctx.start + ctx.settle))                 # tie the threads on here
        F, _rot = MAR._frame(m)
        MAR._tie(prop, m.arm, MAR._ends(m, S.semantic_map(m.arm), F.inverted(), f"perform.{name}.marionette"), F)
        sc.frame_set(keep)
    shapes = _body_shapes(m)
    D = _sample(ctx, m, prop, card, frames, shapes.sources, P["control"])
    packed = shapes.pack()
    src_pos = _unstepped(D["src_pos"], idx)
    src_rot = _orthonormal(_unstepped(D["src_rot"], idx).reshape(-1, 3, 3)).reshape(D["src_rot"].shape)

    def world(i):
        return geom.world(packed, src_rot[i], src_pos[i])

    B = _unstepped(D["B"], idx)
    air = _air(ctx, name, P, ts)
    grav = _gravity(ctx, name, P, ts)
    drag = (P.get("tails") or {}).get("drag", MT.DRAG)
    if not P["control"]:
        return _tails_alone(ctx, m, P, prop, card, frames, ts, idx, shown, D, B, world, air, grav, drag, npts)
    return _set_free(ctx, name, m, P, prop, card, frames, ts, idx, shown, D, B, world, air, grav, drag, npts,
                     held_curves)


def _tails_alone(ctx, m, P, prop, card, frames, ts, idx, shown, D, B, world, air, grav, drag, npts):
    """Six tails on a member with no control: `tails.length` long (default half her height), run over each span she is
    shown in, each settled on its first frame; the control and the whole threads never drawn."""
    n = len(frames)
    for ob in [bpy.data.objects[v] for v in card["bars"].values()] + [bpy.data.objects[f"{prop.name}_thread_{p}"]
                                                                      for p in MR.PARTS]:
        MAR._key_hidden(ob, frames, np.ones(n, bool))
    i0 = int(np.clip(np.searchsorted(frames, ctx.start + ctx.settle), 0, n - 1))
    height = float(D["B"][i0, 0, 2] - 0.5 * (D["B"][i0, 4, 2] + D["B"][i0, 5, 2])) / 0.72   # crown over knees: 72 %
    L = (P["tails"] or {}).get("length") or round(0.5 * height, 3)
    lengths = np.full(6, L)
    pts = np.repeat(B[:, :, None, :], TAIL_N, 2)
    on = shown & (frames >= ctx.frame0)
    runs = []
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    for a, b in zip(edges[0::2], edges[1::2]):
        sl = slice(int(a), int(b))
        if b - a < 2:
            continue
        pts[sl] = _rope(ts, sl, idx, B, lengths, None, [], world, grav, air, drag,
                        settle=(P["tails"] or {}).get("settle", 1.5))
        runs.append([int(frames[a]), int(frames[b - 1])])
    for k, part in enumerate(MR.PARTS):
        _draw(prop, part, "cut", frames, pts[:, k], on & (part in card.get("threads", MR.PARTS)), npts)
    MAR._key_width(P, prop, frames, ts)
    return {"prop": prop.name, "tails": {"length_m": L, "runs": runs, **MT.report(pts[on], lengths)}}


def _scissors(ctx, m, P, frames, ts, idx, D, M):
    """Key the scissors on every frame: lying where they lie, in her hand (their grip where her fingers are, their blades
    along them, the sheet's face to her palm), let go and falling (then lying where they lay again, out of the picture),
    their blades opening as she brings them up and shut ON the snip. Returns (cut point (n, 3) world, hold, report)."""
    W = P["world"]
    hold = MF.scissors_hold(ts, P, M)
    n = len(frames)
    sp = ctx.props.get(W["scissors"]["prop"]) if W.get("scissors") else None
    if sp is None:
        return None, hold, None
    root = sp.root
    card = sp.card
    grips = card.get("use", {}).get("grip", [])
    g = next((u for u in grips if u.get("name") == "hold"), grips[0] if grips else None)
    h_local = Vector(g["point"]) if g and "point" in g else Vector((0.0, 0.0, 0.0))
    looks = {u["name"]: u for u in card.get("use", {}).get("look", [])}
    cut_ob = bpy.data.objects.get((looks.get("cut") or {}).get("object") or f"{sp.name}_cut")
    if cut_ob is not None:
        c_local = root.matrix_world.inverted() @ cut_ob.matrix_world.translation
    elif "cut" in looks and "point" in looks["cut"]:
        c_local = Vector(looks["cut"]["point"])
    else:
        c_local = Vector((0.0, 0.0, 0.035))
    M0 = root.matrix_world.copy()
    side = hold["side"]
    Wr, along, palm = _hand_frame(D, side, _palm_sign(m, side))
    Rh = np.stack([np.cross(-palm, along), -palm, along], axis=2)      # columns x, y, z: the blades along her fingers,
    #                                                                    the face the card's grip has its palm on (-y) hers
    o = Wr + GRIP[0] * along + GRIP[1] * palm - np.einsum("nij,j->ni", Rh, np.array(h_local))
    R0, p0 = np.array(M0.to_3x3().normalized()), np.array(M0.translation)
    rot = np.tile(R0, (n, 1, 1))
    loc = np.tile(p0, (n, 1))
    held = hold["held"]
    take = np.flatnonzero(held & ~np.r_[False, held[:-1]])
    for i in np.flatnonzero(held):
        rot[i], loc[i] = Rh[i], o[i]
    for i0 in take:                                               # into her hand over three frames
        for j, w in enumerate((0.33, 0.67)):
            k = i0 + j
            if k < n and held[k]:
                q = Quaternion(Matrix(R0.tolist()).to_quaternion()).slerp(Matrix(Rh[k].tolist()).to_quaternion(), w)
                rot[k] = np.array(q.to_matrix())
                loc[k] = p0 * (1.0 - w) + o[k] * w
    floor = float(MAR._rest_frame(m)[0].translation.z)
    for t_let, t_end in hold["falling"]:                          # let go: it falls, turning, and lies where it lands
        k = (ts >= t_let) & (ts < t_end)
        ks = np.flatnonzero(k)
        if not len(ks) or ks[0] < 2:
            continue
        a = ks[0] - 1
        v0 = (o[a] - o[max(a - 5, 0)]) / max(ts[a] - ts[max(a - 5, 0)], 1e-6)
        tau = ts[ks] - ts[a]
        loc[ks] = o[a] + np.outer(tau, v0) + np.outer(0.5 * tau * tau, (0.0, 0.0, -MR.GRAVITY))
        spin = [np.array((Matrix.Rotation(5.0 * float(t), 3, Vector((0.3, 1.0, 0.2)).normalized())).to_3x3())
                for t in tau]
        rot[ks] = np.einsum("nij,jk->nik", np.array(spin), Rh[a])
        low = ks[loc[ks, 2] < floor + 0.004]
        if len(low):
            loc[low[0]:ks[-1] + 1] = loc[low[0]] * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, floor + 0.004])
            rot[low[0]:ks[-1] + 1] = rot[low[0]]
    free = ~held                                                  # in her hand they are where it is drawn on every frame
    step = np.where(free & free[idx], idx, np.arange(n))          # (her arms move between her steps); let go, on her
    loc, rot = loc[step], rot[step]                               # steps from the first that begins after it
    cut = loc + np.einsum("nij,j->ni", rot, np.array(c_local))
    Pw = root.parent.matrix_world @ root.matrix_parent_inverse if root.parent else Matrix.Identity(4)
    Pi = np.array(Pw.inverted())
    scale = root.matrix_basis.to_scale()
    root.rotation_mode = "QUATERNION"
    locs, quats = [], []
    for i in range(n):
        Mw = Matrix(rot[i].tolist()).to_4x4()
        Mw.translation = Vector(loc[i])
        Mb = Matrix(Pi.tolist()) @ Mw
        locs.append(tuple(Mb.translation))
        quats.append(tuple(Mb.to_quaternion()))
    K.key_vec(root, "location", frames, locs, interp="LINEAR")
    K.key_vec(root, "rotation_quaternion", frames, K.continuous(quats), interp="LINEAR")
    root.scale = scale
    blades = None
    if "open" in root.keys():
        vals = np.where(held, hold["open"], 0.0)[idx]
        K.set_fcurve(root, '["open"]', 0, frames, vals, interp="LINEAR")
        blades = "open"
    else:
        ctx.log("WARNING", f"perform.{m.name}.marionette.scissors: {sp.name!r} has no `open` root property: its "
                           f"blades are not keyed (a library:paper_scissors has one)")
    return cut, hold, {"prop": sp.name, "held_s": round(float(held.sum()) / ctx.fps, 2), "blades": blades}


def _route_back(Db, i, B, G, part):
    """A thread of hers from its tie `B` down behind her to `G`: round the back of her head (the head's), behind her
    neck, down her back, behind her seat, then on to G."""
    ax = Db["axes"][i]
    back, up = ax[:, 1], ax[:, 2]
    head = Db["body"]["head"][i] + up * 0.09
    way = [B]
    if part == "head":
        way += [head + back * SKULL + up * 0.04, head + back * SKULL - up * 0.06]
    way += [Db["body"]["neck"][i] + back * 0.085, Db["body"]["upper_body2"][i] + back * 0.13,
            Db["body"]["center"][i] + back * 0.15 - up * 0.08, G]
    return np.array(way)


def _set_free(ctx, name, m, P, prop, card, frames, ts, idx, shown, D, B, world, air, grav, drag, npts, held_curves):
    """The threads of a marionette set free (module docstring), the scissors and the snip's empties. Returns the report."""
    Pf = MF.snapped(P, ts)
    W = Pf["world"]
    n = len(frames)
    ph, _t0, since = MF.phases(ts, Pf)
    M = MF.moments(Pf)
    gw = MF._gathered(ts, Pf, M)
    swept = MF._gathered(ts, Pf, M, wait=False)                  # her fist's sweep: the other wrist's waits for its hand
    side_g = W["gather"]["side"] if W.get("gather") else "L"
    Fw, Falong, Fpalm = _hand_frame(D, side_g, _palm_sign(m, side_g))
    fist = D["hand"][("middle1", side_g)] + FIST_IN * Fpalm
    up = D["axes"][:, :, 2]
    cut_sc, hold, sc_rep = _scissors(ctx, m, Pf, frames, ts, idx, D, M)
    X = fist + MF.CUT_ABOVE * up
    if cut_sc is not None:                                        # the cut is between the blades once they are open and
        gap = np.linalg.norm(cut_sc - X, axis=1)                  # at the bundle (on their way up it stays in her fist)
        at = np.clip((BLADES_AT[1] - gap) / (BLADES_AT[1] - BLADES_AT[0]), 0.0, 1.0)
        at = at * (hold["held"] & (hold["open"] > 0.25))
        X = X + at[:, None] * (cut_sc - X)
    A = D["A"]
    held = {p: (MR.held(ts, Pf, p) > 0.5) & (ph == MF.SEATED) for p in MR.PARTS}
    # each thread's own length: its span with the control at home, before her fist comes to it
    lengths = {}
    for j, p in enumerate(MR.PARTS):
        span = np.linalg.norm(A[:, j] - D["B"][:, j], axis=1)
        home = held[p] & (since > 0.25) & (swept[p] < 1e-6)
        lengths[p] = float(span[home].min()) if home.any() else float(span[held[p]].max()) if held[p].any() else 1.0
    main = np.repeat(D["B"][:, :, None, :], npts, 2).copy()
    low = main.copy()
    show_main, show_low = np.zeros((n, 6), bool), np.zeros((n, 6), bool)
    snip_at = D["B"].copy()
    ax = D["axes"]
    for j, p in enumerate(MR.PARTS):
        sd = ax[:, :, 0] * MAR.SIDE[p]
        for i in np.flatnonzero(held[p]):                         # one her fist has come to while its hand is still on
            taut = swept[p][i] > 1e-4                             # its way up runs straight from the control to it
            line = MR.sag_points(A[i, j], D["B"][i, j], 0.0 if taut else lengths[p], npts, sd[i])
            w = gw[p][i]
            if w > 1e-4:
                line = MT.resample(_gathering(D, i, j, p, A[i], X[i], fist[i], w), npts)
            main[i, j] = line
            show_main[i, j] = True
            snip_at[i, j] = X[i]
    rep = {"tails": [], "lengths_m": {p: round(v, 3) for p, v in lengths.items()}}
    # after each snip: the pieces on the falling control and her tails, held in her fist until she lets go
    runs = MF.tail_runs(M)
    snip_runs = {s: b for kind, s, b in runs if kind == "snip"}
    for s, lt in zip(M["snip"], M["let"]):
        if s <= ts[0] or s >= ts[-1]:
            continue
        i_s = int(np.searchsorted(ts, s - 1e-9))
        stop = snip_runs[s]
        i_e = int(min(np.searchsorted(ts, stop), n))
        if i_e - i_s < 3:
            continue
        sl = slice(i_s, i_e)
        a = i_s - 1
        inits_low, inits_up, s_fist, L_low, L_up = [], [], [], [], []
        for j, p in enumerate(MR.PARTS):
            upper, lower = _gathered_path(D, a, j, p, A[a], X[a], fist[a])   # anchor .. cut; fist .. her
            lowp = np.vstack([lower[::-1], [X[a] - PART_GAP * up[a]]])                # her .. fist, below the cut
            upp = np.vstack([upper, [X[a] + PART_GAP * up[a]]])                       # the anchor .. over the cut
            inits_low.append(lowp)
            inits_up.append(upp)
            s_fist.append(MT.arc(lowp[:-1]))
            L_low.append(MT.arc(lowp))
            L_up.append(MT.arc(upp))
        let_on = ts < lt
        Fu = _unstepped(fist, idx)
        tails = _rope(ts, sl, idx, B, L_low, inits_low, [(k, s_fist[k], Fu, let_on) for k in range(6)], world, grav, air,
                      drag)
        for j in range(6):                                       # the pieces on the control whip up through their eyes
            for i in range(sl.start, sl.stop):
                tau = ts[idx[i]] - s
                main[i, j] = MT.resample(_whip(inits_up[j], A[i, j] - A[a, j], tau), npts)
        for j in range(6):
            low[sl, j] = np.array([MT.resample(q, npts) for q in tails[:, j]])
            show_low[sl, j] = True
            show_main[sl, j] = ts[sl] < s + MF.SHOWN_AFTER_SNIP
            snip_at[sl, j] = tails[:, j, -1]
        rep["tails"].append({"snip": round(s, 3), "frames": [int(frames[i_s]), int(frames[i_e - 1])],
                             "lengths_m": [round(x, 3) for x in L_low], **MT.report(tails, L_low)})
    # over each haul: her tails from before the catch, gathered at the hand, let go as she falls into her seat
    G = _unstepped(D["grab"], idx)
    for (h, to), (_kind, t_from, t_end) in zip(zip(M["haul"], M["to"]), [r for r in runs if r[0] == "haul"]):
        if t_end <= max(ts[0], 0.0) or h >= ts[-1]:
            continue
        i_a = int(np.searchsorted(ts, t_from))
        i_c = int(min(np.searchsorted(ts, h - 1e-9), n - 1))
        i_e = int(min(np.searchsorted(ts, t_end - 1e-9), n))
        if i_e - i_a < 3:
            continue
        sl = slice(i_a, i_e)
        inits, L = [], []
        for j, p in enumerate(MR.PARTS):
            way = (_route_back(D, i_c, D["B"][i_c, j], G[i_c], p) if p in ("head", "back")
                   else np.array([D["B"][i_c, j], G[i_c]]))
            L.append(MT.arc(way))
            inits.append(_route_back(D, i_a, D["B"][i_a, j], G[i_a], p) if p in ("head", "back")
                         else np.array([D["B"][i_a, j], G[i_a]]))
        pin_on = (ts >= h - MF.APPROACH) & (ts < to)
        tails = _rope(ts, sl, idx, B, L, inits, [(k, L[k], G, pin_on) for k in range(6)], world, grav, air, drag,
                      settle=0.5)
        vis = ts[sl] >= h
        for j in range(6):
            low[sl, j] = np.array([MT.resample(q, npts) for q in tails[:, j]])
            show_low[sl, j] = vis
            snip_at[sl, j] = tails[:, j, -1]
        rep["tails"].append({"haul": round(h, 3), "frames": [int(frames[i_a]), int(frames[i_e - 1])],
                             "lengths_m": [round(x, 3) for x in L], **(MT.report(tails[vis], L) if vis.any() else {})})
    Ri = np.array(prop.root.matrix_world.inverted())
    exist = set(card.get("threads", MR.PARTS))                  # the prop's `threads`: the others never drawn
    for j, p in enumerate(MR.PARTS):
        for kind, pts, vis in (("thread", main[:, j], show_main[:, j]), ("cut", low[:, j], show_low[:, j])):
            _draw(prop, p, kind, frames, pts, vis & shown & (p in exist), npts)
        sn = bpy.data.objects[f"{prop.name}_snip_{p}"]
        K.key_vec(sn, "location", frames, [tuple(v) for v in snip_at[:, j] @ Ri[:3, :3].T + Ri[:3, 3]], interp="LINEAR")
    for key, v in card["bars"].items():                          # the control only while she is shown, the leg bar
        ob = bpy.data.objects[v]                                 # never without knee threads
        fc = ob.animation_data.action.fcurves.find("hide_render") if ob.animation_data and ob.animation_data.action \
            else None
        was = np.array([fc.evaluate(float(f)) for f in frames]) >= 0.5 if fc else np.zeros(n, bool)
        gone = key == "leg" and not exist & {"knee_L", "knee_R"}
        MAR._key_hidden(ob, frames, was | ~shown | gone)
    MAR._key_width(P, prop, frames, ts)
    rep.update({"prop": prop.name, "frames": [int(frames[0]), int(frames[-1])], "bar_curves_stepped": held_curves,
                "scissors": sc_rep, "threads": sorted(exist)})
    return rep


def _whip(poly, moved, tau, out=0.12):
    """A cut piece on the control (`poly`, its anchor first, as it was at the cut; the anchor since `moved`) `tau` s after
    the cut: under tension it whips up through its eye and out above it within `out` s (the thread a distance s along
    it from the eye lies s - D(tau) along its old line, or straight up above the eye once that is negative)."""
    P = np.asarray(poly, float) + moved
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    D = (cum[-1] + 0.6) * (max(tau, 0.0) / out) ** 1.6
    s = np.linspace(0.0, cum[-1], 24) - D
    along = np.stack([np.interp(np.maximum(s, 0.0), cum, P[:, k]) for k in range(3)], 1)
    along[s < 0.0] = P[0] + np.outer(-s[s < 0.0], (0.0, 0.0, 1.0))
    return along


def _gathered_path(D, i, j, part, A, X, fist):
    """(upper [anchor, the bundle's top], lower [fist, ..., her]) of thread j gathered into her fist on frame i (the
    cut point X between them): from its anchor (A, all six anchors (6, 3)) to the top of the bundle (BUNDLE_UP over the
    cut, toward the control: the six run up together from her fist, then part to their sticks), the cut point, her
    fist, then (the back's) over the top of her head, down behind it and behind her neck (never round her face: cut,
    its piece floats up from behind her), and down to where it meets her."""
    up = MT._unit(A.mean(axis=0) - X)
    upper = np.array([A[j], X + BUNDLE_UP * up])
    lower = [fist]
    if part == "back":
        ax = D["axes"][i]
        back, upw = ax[:, 1], ax[:, 2]
        head = D["body"]["head"][i] + upw * 0.09
        lower += [head + upw * 0.1 + back * 0.03, head + back * SKULL + upw * 0.02,
                  D["body"]["neck"][i] + back * 0.085]
    lower.append(D["B"][i, j])
    return upper, np.array(lower)


def _gathering(D, i, j, part, A, X, fist, w):
    """The polyline of thread j a share `w` gathered: its waypoints (`_gathered_path`) each moved from the point of its
    own line nearest her fist (where the fist catches it) to their places: at 0 the thread's line, at 1 gathered."""
    upper, lower = _gathered_path(D, i, j, part, A, X, fist)
    a, b = upper[0], lower[-1]
    ab = b - a
    c0 = a + float(np.clip((fist - a) @ ab / max(float(ab @ ab), 1e-9), 0.0, 1.0)) * ab
    inner = np.vstack([upper[1:], [X], lower[:-1]])
    return np.vstack([[a], c0 + w * (inner - c0), [b]])
