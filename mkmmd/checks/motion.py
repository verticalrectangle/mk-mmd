"""Metrics on sampled motion: jitter, contact, penetration, clearance, foot_slide, joint_limits."""
import numpy as np

from ..core import families as FAM
from ..solvers import geom
from . import CheckError, Metric, metric

STATIC_MM = 0.01          # median speed (mm/frame) below which a chain counts as still for the jitter ratio


def _list(v):
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def _peaks(values, frames, n=5):
    order = np.argsort(-np.asarray(values))[:n]
    return sorted(int(frames[i]) for i in order)


# ---------------------------------------------------------------- jitter
@metric
class Jitter(Metric):
    name = "jitter"
    doc = ("Shake of secondary motion: bone points (heads; tails of chain ends) in the frame of a reference bone, "
           "median frame-to-frame change of velocity ('jerk') over median speed (mm per frame). Calm hair that "
           "follows the body: < 0.5; Bullet hair on a seated character: ~1.4. Use consecutive frames.")
    args = {"family": "chain family or list (hair families: " + ", ".join(sorted(FAM.HAIR_FAMILIES)) + ", ears, "
                      "tail, skirt, ...)",
            "bones": "explicit bone names instead of / besides families",
            "ref": "reference bone (default head)", "cast": "cast member (or armature)"}

    def needs(self, args, ctx, need):
        arm, _ = ctx.cast(args)
        fams, bones = _list(args.get("family") or args.get("families")), _list(args.get("bones"))
        if not fams and not bones:
            raise CheckError("jitter: give family or bones")
        if fams:
            bad = [f for f in fams if f not in FAM.FAMILIES]
            if bad:
                raise CheckError(f"jitter: unknown families {bad} (have {FAM.FAMILIES})")
            need.family(arm, *fams)
        need.bone(arm, *bones, args.get("ref", "head"))
        return {"arm": arm, "fams": fams, "bones": bones}

    def compute(self, args, ctx, data, frames, st):
        arm = st["arm"]
        names = list(dict.fromkeys(data.family(arm, st["fams"]) + data.resolve(arm, st["bones"])))
        if not names:
            raise CheckError(f"jitter: no bones in families {st['fams']}")
        if len(frames) < 4:
            raise CheckError("jitter: needs at least 4 frames")
        W, _, names = data.bones(arm, names, frames)
        Rw, _, _ = data.bones(arm, [args.get("ref", "head")], frames)
        sel = set(names)
        leaf = np.array([not any(c in sel for c in data.info(arm, n)["children"]) for n in names])
        L = np.array([data.info(arm, n)["length"] for n in names])
        pts = W[..., :3, 3].copy()
        tails = geom.apply(W, np.stack([np.zeros_like(L), L, np.zeros_like(L)], 1)[None])
        pts[:, leaf] = tails[:, leaf]
        inv = np.linalg.inv(Rw[:, 0])
        P = geom.apply(inv[:, None], pts) * 1000.0
        jerk = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=2)
        speed = np.linalg.norm(P[1:] - P[:-1], axis=2)
        jm, sm = float(np.median(jerk)), float(np.median(speed))
        ratio = jm / max(sm, 1e-6)
        fps = ctx.project.fps if ctx.project else 30.0
        per_s = [round(float(np.percentile(jerk[a:a + int(fps)], 99)), 1) for a in range(0, len(jerk), int(fps))]
        worst = [names[i] for i in np.argsort(-np.percentile(jerk, 95, axis=0))[:3]]
        detail = {"bones": len(names), "jerk_median_mm": round(jm, 4), "speed_median_mm": round(sm, 4),
                  "jerk_p95_mm": round(float(np.percentile(jerk, 95)), 2), "jerk_max_mm": round(float(jerk.max()), 1),
                  "at_frame": int(np.asarray(frames)[1:-1][int(np.argmax(jerk.max(1)))]),
                  "peak_frames": _peaks(jerk.max(1), frames[1:-1]), "worst_bones": worst,
                  "jerk_p99_per_second_mm": per_s}
        if sm < STATIC_MM:
            detail["static"] = (f"median speed {sm:.4f} mm/frame: the chain is essentially still and the ratio sits at "
                                f"Blender's float32 noise; judge it by jerk_p95_mm / peak_frames instead")
        return ratio, detail


# ---------------------------------------------------------------- contact
UNITS = {"m": 1.0, "cm": 100.0, "mm": 1000.0}


def _side(ctx, data, ref, frames, st_idx):
    if isinstance(ref, str) and ref.startswith("track:"):
        return ctx.track(ref, frames)
    return data.expr(st_idx, frames)


@metric
class Contact(Metric):
    name = "contact"
    doc = ("Distance between two tracked points per frame: a nib and its ink path, a hand and a wheel rim, feet and "
           "pedals. Value: the largest |distance| (or |difference| along one axis) over the frames where `when` holds.")
    args = {"a": "expression for point A (`mk q` syntax) or track:NAME.channel",
            "b": "expression or track:NAME.channel for point B",
            "when": "optional mask: track:NAME.channel or expression (truthy frames count)",
            "component": "x | y | z: signed difference along one world axis instead of the distance",
            "unit": "m | cm | mm (default mm)"}

    def needs(self, args, ctx, need):
        st = {}
        for k in ("a", "b", "when"):
            v = args.get(k)
            if v is None:
                if k != "when":
                    raise CheckError(f"contact: give {k}")
                continue
            if not (isinstance(v, str) and v.startswith("track:")):
                st[k] = need.expr(v)
        if args.get("unit", "mm") not in UNITS:
            raise CheckError(f"contact: unit must be one of {list(UNITS)}")
        return st

    def compute(self, args, ctx, data, frames, st):
        A = np.asarray(_side(ctx, data, args["a"], frames, st.get("a")), float).reshape(len(frames), -1)
        B = np.asarray(_side(ctx, data, args["b"], frames, st.get("b")), float).reshape(len(frames), -1)
        mask = np.ones(len(frames), bool)
        if args.get("when") is not None:
            w = np.asarray(_side(ctx, data, args["when"], frames, st.get("when")), float).reshape(len(frames))
            mask = np.nan_to_num(w, nan=0.0) > 0.5
        d = A - B
        mask &= ~np.isnan(d).any(1)
        if not mask.any():
            raise CheckError("contact: no frames where `when` holds and both points exist")
        comp = args.get("component")
        v = d[:, "xyz".index(comp)] if comp else np.linalg.norm(d, axis=1)
        v = v * UNITS[args.get("unit", "mm")]
        f = np.asarray(frames)[mask]
        vm = v[mask]
        a = np.abs(vm)
        return float(a.max()), {"frames": int(mask.sum()), "median": round(float(np.median(vm)), 3),
                                "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3),
                                "at_frame": int(f[int(np.argmax(a))]), "worst_frames": _peaks(a, f),
                                "unit": args.get("unit", "mm")}


# ---------------------------------------------------------------- penetration
@metric
class Penetration(Metric):
    name = "penetration"
    doc = ("How deep chain bones (hair, ears, tails, skirts) sink into the model's own collision bodies and the "
           "scene's colliders, measured like the mk secondary-motion solver: points along each segment with the "
           "chain's radius; rest-pose overlaps and the region near each root are exempt. Value: deepest mm.")
    args = {"cast": "cast member (needs its rig.json)", "families": "chain families (default: hair + ears)",
            "radius": "{family: {scale, max}}: point radius = body radius x scale, at most max (m); "
                      "or one {scale, max} for all",
            "colliders": "scene collider specs or the name of a [colliders] set",
            "model_bodies": "collide with the model's own bodies (default true)",
            "use_masks": "honour PMX collision masks everywhere, not only near the root (default false)",
            "anchor_free": "metres near each root free of root/mask collisions (default 0.25)"}

    def needs(self, args, ctx, need):
        arm, rig = ctx.cast(args)
        if rig is None:
            raise CheckError("penetration: needs the model's rig.json (cast with rig/asset, or rig = path)")
        fams = _list(args.get("families")) or sorted(FAM.HAIR_FAMILIES | {"ears"})
        rig = dict(rig, chains=[c for _, c in geom.split_chains(rig, fams)])     # as the sim stage solves them
        chains = geom.Chains(rig, fams)
        if not len(chains):
            raise CheckError(f"penetration: no chains in families {fams}")
        need.bone(arm, *chains.bones)
        if args.get("model_bodies", True):
            need.bone(arm, *[b["bone"] for b in rig["bodies"] if b.get("bone") and "geom" in b and
                             b["bone"] not in chains.chain_bones])
        specs = ctx.colliders(args.get("colliders"))
        sim_params = {}
        if ctx.project and args.get("cast"):
            sim_params = (ctx.project.data.get("sim", {}).get(args["cast"]) or {}).get("params") or {}
        from ..solvers.strands import resolve_params
        params = resolve_params(sorted(set(chains.family)), sim_params)
        return {"arm": arm, "rig": rig, "chains": chains, "idx": need.collider_specs(specs), "params": params}

    def compute(self, args, ctx, data, frames, st):
        arm, rig, chains = st["arm"], st["rig"], st["chains"]
        shapes = geom.Shapes()
        if args.get("model_bodies", True):
            geom.model_shapes(shapes, rig, arm, skip_bones=chains.chain_bones)
        for it in data.colliders(st["idx"]):
            shapes.add_item(it)
        P = shapes.pack()
        labels = []
        for kind in geom.KINDS:
            for k in range(P[kind]["n"]):
                m = P[kind]["model"][k]
                labels.append(f"body:{rig['bodies'][m]['bone']}" if m is not None else P[kind]["tag"][k])
        labels.append("floor")
        rad_arg = args.get("radius")
        if rad_arg:                                      # explicit: {scale, max} or {family: {scale, max}}
            per = rad_arg if any(isinstance(v, dict) for v in rad_arg.values()) else \
                {f: rad_arg for f in set(chains.family)}
            scale = np.array([per.get(f, {}).get("scale", 1.0) for f in chains.family], float)
            rmax = np.array([per.get(f, {}).get("max", np.inf) for f in chains.family], float)
        else:                                            # what the solver used: [sim.<cast>] params over its defaults
            pr = st["params"]
            scale = np.array([pr[f]["radius"] for f in chains.family], float)
            rmax = np.array([pr[f]["radius_max"] for f in chains.family], float)
        rad = np.minimum(chains.radius * scale, rmax)
        Rs, ps, R0, p0 = data.source_frames(shapes.sources, frames)
        rest_W = geom.world(P, R0, p0)
        enable = geom.enable_matrix(chains, P, rig, rest_W, radius=rad, use_masks=bool(args.get("use_masks")),
                                    anchor_free=float(args.get("anchor_free", geom.ANCHOR_FREE)))
        X0, own, _ = chains.points()
        rr = rad[own]
        W, rest, _ = data.bones(arm, chains.bones, frames)
        D = W @ np.linalg.inv(rest)[None]                       # rest (armature space) -> posed world, per bone
        best = np.zeros(len(frames))
        worst = []
        deep = {}
        for k in range(len(frames)):
            X = geom.apply(D[k][own], X0)
            Wk = geom.world(P, Rs[k], ps[k])
            pen, sid = geom.measure(X, rr, Wk, shapes.floor_z, P, enable, own)
            j = int(np.argmax(pen))
            best[k] = pen[j]
            label = labels[sid[j]] if sid[j] >= 0 else None
            worst.append((chains.bones[own[j]], label))
            if pen[j] > 0.002:
                deep[label] = deep.get(label, 0) + 1
        k = int(np.argmax(best))
        mm = best * 1000.0
        return float(max(mm[k], 0.0)), {
            "at_frame": int(frames[k]), "bone": worst[k][0], "into": worst[k][1],
            "p95_mm": round(float(np.percentile(np.maximum(mm, 0), 95)), 3),
            "frames_over_2mm_by_shape": dict(sorted(deep.items(), key=lambda kv: -kv[1])[:5]),
            "points": int(len(X0)), "shapes": len(labels) - 1}


# ---------------------------------------------------------------- clearance
SEARCH_ITERS = 60          # ternary steps along a capsule's axis: the interval shrinks to (2/3)^60 ~ 3e-11 of it
SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)        # a box's 8 corners
EDGES = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)
                  if np.abs(SIGNS[i] - SIGNS[j]).sum() == 2])                                 # and its 12 edges


def _to_boxes(P, lo, hi):
    """Distance from points P (m, ..., 3) to the boxes lo..hi (m, 3)."""
    sh = (len(lo),) + (1,) * (P.ndim - 2) + (3,)
    return np.linalg.norm(P - np.clip(P, lo.reshape(sh), hi.reshape(sh)), axis=-1)


def _seg_seg(p1, q1, p2, q2):
    """Least distance between the segments p1-q1 and p2-q2 (broadcasting arrays of points; Ericson, Real-Time
    Collision Detection 5.1.9, points as segments of length 0 included)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    ok1, ok2 = a > 1e-18, e > 1e-18
    sa, se = np.where(ok1, a, 1.0), np.where(ok2, e, 1.0)
    den = a * e - b * b
    s = np.where(den > 1e-12 * sa * se, np.clip((b * f - c * e) / np.where(den > 0, den, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / se
    tc = np.clip(t, 0.0, 1.0)
    s = np.where(t != tc, np.clip((b * tc - c) / sa, 0.0, 1.0), s)
    s, t = (np.where(ok1 & ok2, s, np.where(ok1, np.clip(-c / sa, 0.0, 1.0), 0.0)),
            np.where(ok1 & ok2, tc, np.where(ok2, np.clip(f / se, 0.0, 1.0), 0.0)))
    return np.linalg.norm(p1 + d1 * s[..., None] - p2 - d2 * t[..., None], axis=-1)


def _capsule_gaps(A, B, lo, hi):
    """Least distance (m, K) from the segments A-B (K, 3) to the boxes lo..hi (m, 3): the distance from a point of a
    segment to a box is convex along it, so a ternary search finds its minimum."""
    a, b = np.zeros((len(lo), len(A))), np.ones((len(lo), len(A)))

    def gap(t):                                                                           # (m, K) at shares t of each
        return _to_boxes(A[None] + t[..., None] * (B - A)[None], lo, hi)
    for _ in range(SEARCH_ITERS):
        t1, t2 = a + (b - a) / 3.0, b - (b - a) / 3.0
        left = gap(t1) <= gap(t2)                                                         # the minimum is in [a, t2]
        a, b = np.where(left, a, t1), np.where(left, t2, b)
    return gap((a + b) / 2.0)


def _box_gaps(C, Rk, h, lo, hi):
    """Least distance (m, K) between the boxes C, Rk (K, 3), (K, 3, 3) of half sizes h (K, 3) and the axis-aligned
    boxes lo..hi (m, 3); 0 where they overlap. Between two disjoint convex polyhedra it is reached from a corner of
    one to the other or between an edge of each; they overlap when no separating axis exists (the 3 + 3 face normals
    and the 9 cross products of their edges)."""
    V = C[:, None] + np.einsum("kij,knj->kni", Rk, SIGNS[None] * h[:, None])              # (K, 8, 3) corners
    U = (lo + hi)[:, None] / 2.0 + SIGNS[None] * ((hi - lo) / 2.0)[:, None]               # (m, 8, 3)
    d = _to_boxes(np.broadcast_to(V[None], (len(lo),) + V.shape), lo, hi).min(2)
    loc = np.einsum("kji,mknj->mkni", Rk, U[:, None] - C[None, :, None])                  # U in each box's frame
    d = np.minimum(d, np.linalg.norm(loc - np.clip(loc, -h[None, :, None], h[None, :, None]), axis=-1).min(2))
    ev, eu = V[:, EDGES], U[:, EDGES]                                                     # (K, 12, 2, 3), (m, 12, 2, 3)
    d = np.minimum(d, _seg_seg(ev[None, :, :, None, 0], ev[None, :, :, None, 1],
                               eu[:, None, None, :, 0], eu[:, None, None, :, 1]).min((2, 3)))
    ax = np.concatenate([np.broadcast_to(np.eye(3)[None], Rk.shape), Rk.transpose(0, 2, 1),
                         np.cross(np.eye(3)[None, :, None], Rk.transpose(0, 2, 1)[:, None]).reshape(-1, 9, 3)], 1)
    n = np.linalg.norm(ax, axis=2, keepdims=True)
    ax = np.where(n > 1e-9, ax / np.maximum(n, 1e-12), 0.0)                               # (K, 15, 3); 0: parallel edges
    gap = np.abs(np.einsum("mki,kai->mka", C[None] - ((lo + hi) / 2.0)[:, None], ax))
    ra = np.einsum("mi,kai->mka", (hi - lo) / 2.0, np.abs(ax))
    rb = np.einsum("kj,kaj->ka", h, np.abs(np.einsum("kai,kij->kaj", ax, Rk)))[None]
    apart = (gap > ra + rb + 1e-12).any(2)
    return np.where(apart, d, 0.0)


def body_gaps(W, lo, hi):
    """Gaps (m, K) in metres between m axis-aligned boxes lo..hi (m, 3) and every shape of world geometry `W`
    (geom.world of a model's own bodies, the K shapes in labels order): the least distance from the box to the shape's
    core less its radius (a sphere's or capsule's, a box's rounding), exact. Negative: the shape reaches into the box
    (when their cores overlap the gap is minus the radius, not the depth)."""
    if "cylinder" in W:
        raise CheckError("clearance: a cylinder is a scene collider, never a model's body")
    out = []
    if "sphere" in W:
        Cs, Rs = W["sphere"]
        out.append(_to_boxes(np.broadcast_to(Cs[None], (len(lo),) + Cs.shape), lo, hi) - Rs[None])
    if "capsule" in W:
        A, B, Rc = W["capsule"]
        out.append(_capsule_gaps(A, B, lo, hi) - Rc[None])
    if "box" in W:
        C, Rk, half, rnd = W["box"]
        out.append(_box_gaps(C, Rk, np.maximum(half - rnd[:, None], 0.0), lo, hi) - rnd[None])
    return np.concatenate(out, 1) if out else np.zeros((len(lo), 0))


@metric
class Clearance(Metric):
    name = "clearance"
    doc = ("How close things that fly past a character come to it: the world bounds of each object named (names or "
           "fnmatch patterns, e.g. a lyric entry's words `words_l*`), on the frames it has geometry, against the "
           "character's own collision bodies (rig.json). Value: the least gap over frames and objects, metres "
           "(negative: into a body); detail: which object, on which frame, nearest which body.")
    args = {"cast": "cast member (needs its rig.json)", "objects": "object names or fnmatch patterns (required)"}
    default_min = 0.0

    def needs(self, args, ctx, need):
        arm, rig = ctx.cast(args)
        if rig is None:
            raise CheckError("clearance: needs the model's rig.json (cast with rig/asset, or rig = path)")
        pats = _list(args.get("objects"))
        if not pats:
            raise CheckError("clearance: give `objects` (names or fnmatch patterns of the things that fly past)")
        need.obj(*pats)
        need.bone(arm, *[b["bone"] for b in rig["bodies"] if b.get("bone") and "geom" in b])
        return {"arm": arm, "rig": rig, "pats": pats}

    def compute(self, args, ctx, data, frames, st):
        from fnmatch import fnmatchcase
        arm, rig = st["arm"], st["rig"]
        names = [n for n in data.meta["objects"] if any(fnmatchcase(n, p) for p in st["pats"])]
        if not names:
            raise CheckError(f"clearance: no object matches {st['pats']}")
        shapes = geom.Shapes()
        geom.model_shapes(shapes, rig, arm)
        P = shapes.pack()
        labels = [f"body:{rig['bodies'][P[k]['model'][i]]['bone']}" for k in geom.KINDS for i in range(P[k]["n"])]
        if not labels:
            raise CheckError(f"clearance: {arm} has no collision bodies with a bone in its rig.json")
        Rs, ps, _, _ = data.source_frames(shapes.sources, frames)
        cols = [data.meta["objects"].index(n) for n in names]
        boxes = data.z["obj_bbox"][data.rows(frames)][:, cols]                 # (F, m, 2, 3)
        live = (boxes[:, :, 1] - boxes[:, :, 0]).max(2) > 0               # (F, m): the object has geometry then
        if not live.any():
            raise CheckError(f"clearance: none of {len(names)} object(s) has geometry on the frames checked")
        best, where = np.inf, None
        for k in np.flatnonzero(live.any(1)):
            js = np.flatnonzero(live[k])
            g = body_gaps(geom.world(P, Rs[k], ps[k]), boxes[k, js, 0], boxes[k, js, 1])
            j, i = np.unravel_index(int(np.argmin(g)), g.shape)
            if g[j, i] < best:
                best, where = float(g[j, i]), (names[js[j]], int(frames[k]), labels[i])
        return round(best, 4), {"object": where[0], "at_frame": where[1], "body": where[2], "objects": len(names),
                                "object_frames": int(live.sum())}


# ---------------------------------------------------------------- foot slide
@metric
class FootSlide(Metric):
    name = "foot_slide"
    doc = ("Feet sliding while planted: horizontal speed of ankles/toes on frames where they rest near the floor "
           "and barely move vertically. Value: the 95th percentile speed in mm per frame over planted frames. Speeds "
           "are taken between consecutive frames only, so `frames` may have gaps (the stops of a dance, say).")
    args = {"cast": "cast member (or armature)", "points": "bones (default ankle.L ankle.R toe.L toe.R)",
            "floor": "floor height (m, default 0)", "tolerance": "how far above its rest height a point still "
                                                                 "counts as planted (m, default 0.015)"}

    def needs(self, args, ctx, need):
        arm, _ = ctx.cast(args)
        pts = _list(args.get("points")) or ["ankle.L", "ankle.R", "?toe.L", "?toe.R"]
        need.bone(arm, *pts)
        return {"arm": arm, "pts": pts}

    def compute(self, args, ctx, data, frames, st):
        W, rest, names = data.bones(st["arm"], st["pts"], frames)
        A = data.arm_world(st["arm"], frames)
        pos = W[..., :3, 3]
        rest_w = geom.apply(A[0][None], rest[:, :3, 3])        # rest heights in world, armature at frame 0
        floor = float(args.get("floor", 0.0))
        tol = float(args.get("tolerance", 0.015))
        hz = rest_w[:, 2] - floor
        f = np.asarray(frames)
        adjacent = np.diff(f) == 1                              # a gap in `frames` is not a frame's motion
        runs = np.split(np.arange(len(f)), np.flatnonzero(~adjacent) + 1)
        out, worst, at = {}, 0.0, None
        for j, n in enumerate(names):
            z = pos[:, j, 2] - floor
            vz = np.zeros(len(z))
            for r in runs:
                if len(r) > 1:
                    vz[r] = np.abs(np.gradient(z[r])) * 1000.0
            planted = (z < hz[j] + tol) & (vz < 2.0)
            h = np.linalg.norm(np.diff(pos[:, j, :2], axis=0), axis=1) * 1000.0
            pl = planted[1:] & planted[:-1] & adjacent
            if pl.sum() < 3:
                out[n] = {"planted_frames": int(pl.sum())}
                continue
            p95 = float(np.percentile(h[pl], 95))
            if p95 >= worst:
                worst, at = p95, int(np.asarray(frames)[1:][int(np.argmax(np.where(pl, h, 0.0)))])
            out[n] = {"planted_frames": int(pl.sum()), "p95_mm_per_frame": round(p95, 3),
                      "max_mm_per_frame": round(float(h[pl].max()), 3),
                      "worst_frames": _peaks(np.where(pl, h, 0.0), frames[1:])}
        out["at_frame"] = at
        return worst, out


# ---------------------------------------------------------------- joint limits
def _signed(t, s, b):
    """Signed bend of segment s relative to t toward direction b (deg), and its sideways deviation (deg)."""
    tn = t / np.linalg.norm(t, axis=1, keepdims=True)
    sn = s / np.linalg.norm(s, axis=1, keepdims=True)
    bp = b - np.sum(b * tn, 1, keepdims=True) * tn
    bp /= np.maximum(np.linalg.norm(bp, axis=1, keepdims=True), 1e-9)
    h = np.cross(tn, bp)
    bend = np.degrees(np.arctan2(np.sum(sn * bp, 1), np.sum(sn * tn, 1)))
    side = np.degrees(np.arcsin(np.clip(np.sum(sn * h, 1), -1, 1)))
    return bend, side


def _swing_twist(R, axis):
    """Swing angle (deg) of rotations R (F,3,3) away from per-frame unit axes (F,3) and twist (deg) about them."""
    q = geom.mat_to_quat(R)
    proj = np.sum(q[:, 1:] * axis, 1)
    tw = np.degrees(2 * np.arctan2(np.abs(proj), np.abs(q[:, 0])))
    tw = np.where(tw > 180, 360 - tw, tw)
    moved = np.einsum("fij,fj->fi", R, axis)
    sw = np.degrees(np.arccos(np.clip(np.sum(moved * axis, 1), -1, 1)))
    return sw, tw


LIMITS = {"elbow_fold": (0.0, 170.0), "elbow_back": (0.0, 20.0), "knee_bend": (-8.0, 165.0),
          "knee_side": (-20.0, 20.0), "wrist": (0.0, 110.0), "neck_swing": (0.0, 75.0), "neck_twist": (0.0, 85.0),
          "spine_swing": (0.0, 75.0), "spine_twist": (0.0, 60.0)}


@metric
class JointLimits(Metric):
    name = "joint_limits"
    doc = ("Anatomy: elbows do not fold through themselves or bend backward, knees bend one way and not sideways, "
           "wrists, neck and spine stay in ranges professional MMD motions keep to. Angles come from the pose "
           "(bones rotated relative to their rest pose; MMD models face -Y at rest). Value: the worst excess over a "
           "limit in degrees (0 = all inside).")
    args = {"cast": "cast member (or armature)", "limits": "override {name: [lo, hi]}: " + ", ".join(LIMITS),
            "sides": "L R (default both)"}

    def needs(self, args, ctx, need):
        arm, _ = ctx.cast(args)
        sides = _list(args.get("sides")) or ["L", "R"]
        names = ["lower_body", "upper_body", "?upper_body2", "neck", "head"]
        for s in sides:
            names += [f"arm.{s}", f"?arm_twist.{s}", f"elbow.{s}", f"wrist.{s}", f"?middle1.{s}", f"leg.{s}",
                      f"knee.{s}", f"ankle.{s}"]
        need.bone(arm, *names)
        return {"arm": arm, "sides": sides}

    def compute(self, args, ctx, data, frames, st):
        arm = st["arm"]
        lim = dict(LIMITS)
        lim.update({k: tuple(v) for k, v in (args.get("limits") or {}).items()})
        A = data.arm_world(arm, frames)[:, :3, :3]
        A = A / np.linalg.norm(A, axis=1, keepdims=True)

        def get(n):
            W, rest, _ = data.bones(arm, [n], frames)
            Rw = geom.unscaled(W[:, 0])[:, :3, :3]
            R0 = geom.unscaled(rest[0])[:3, :3]
            delta = Rw @ R0.T @ np.linalg.inv(A)                  # world rotation of the bone since rest
            return W[:, 0, :3, 3], delta

        fwd = np.einsum("fij,j->fi", A, np.array([0.0, -1.0, 0.0]))
        series = {}
        for s in st["sides"]:
            sh, _ = get(f"arm.{s}")
            el, _ = get(f"elbow.{s}")
            wr, _ = get(f"wrist.{s}")
            hinge = f"arm_twist.{s}" if data.has(arm, f"?arm_twist.{s}") else f"arm.{s}"
            _, dh = get(hinge)
            b = np.einsum("fij,fj->fi", dh, fwd)
            t, u = el - sh, wr - el
            cosf = np.sum(t * u, 1) / (np.linalg.norm(t, axis=1) * np.linalg.norm(u, axis=1) + 1e-12)
            series[f"elbow_fold.{s}"] = np.degrees(np.arccos(np.clip(cosf, -1, 1)))
            # hyperextension only where the bend is clearly in the plane of the upper arm's front: MMD motions often
            # put upper-arm rotation into the elbow (a ball joint in MMD rigs), so out-of-plane bends are normal
            bend, side = _signed(t, u, b)
            series[f"elbow_back.{s}"] = np.where((np.abs(side) < 30) & (bend < 0) & (bend > -90), -bend, 0.0)
            if data.has(arm, f"?middle1.{s}"):
                mid, _ = get(f"middle1.{s}")
                f1, f2 = wr - el, mid - wr
                cosang = np.sum(f1 * f2, 1) / (np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1) + 1e-12)
                series[f"wrist.{s}"] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            hip, dl = get(f"leg.{s}")
            kn, _ = get(f"knee.{s}")
            an, _ = get(f"ankle.{s}")
            back = np.einsum("fij,fj->fi", dl, -fwd)
            kb, ks = _signed(kn - hip, an - kn, back)
            series[f"knee_bend.{s}"], series[f"knee_side.{s}"] = kb, ks
        _, d_up = get("upper_body2" if data.has(arm, "?upper_body2") else "upper_body")
        _, d_head = get("head")
        _, d_low = get("lower_body")
        _, d_ub = get("upper_body")
        up = np.einsum("fij,j->fi", A, np.array([0.0, 0.0, 1.0]))   # the model's rest up, in world, per frame
        rel = np.einsum("fji,fjk->fik", d_up, d_head)            # head relative to the chest
        series["neck_swing"], series["neck_twist"] = _swing_twist(rel, up)
        rel = np.einsum("fji,fjk->fik", d_low, d_ub)
        series["spine_swing"], series["spine_twist"] = _swing_twist(rel, up)
        worst, detail, at = 0.0, {}, None
        for key, v in series.items():
            lo, hi = lim[key.split(".")[0]]
            over = np.maximum(v - hi, 0) + np.maximum(lo - v, 0)
            entry = {"min": round(float(v.min()), 1), "max": round(float(v.max()), 1), "limits": [lo, hi]}
            if over.max() > 0:
                entry["over_deg"] = round(float(over.max()), 1)
                entry["frames"] = [int(f) for f in np.asarray(frames)[over > 0][:10]]
                if float(over.max()) > worst:
                    worst, at = float(over.max()), (key, int(np.asarray(frames)[int(np.argmax(over))]))
            detail[key] = entry
        detail["at_frame"] = at[1] if at else None
        detail["worst"] = at[0] if at else None
        return worst, detail
