"""strum and prop_body: checks for a character playing a worn prop (a guitar): does the pick meet the strings on the beat, and
does the prop stay out of the body (docs/design.md: Checks).

`strum` replays the project's own strum spec (`[perform.<cast>] strum`, mkmmd.core.strum.plan) on the timeline and measures the
scene against it: where the pick tip really is (the pick object rides the wrist, so the arm's reach is in the numbers) in the
frame of the prop, against the strings of the card's `use.grip` entries. `prop_body` sinks the prop's own geometry into the
character's collision bodies (rig.json, the shapes the hair collides with)."""
import json

import numpy as np

from ..core import fretting as FR
from ..core import strum as SM
from ..solvers import geom
from . import CheckError, Metric, metric


def _list(v):
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def _specs(ctx, args):
    """The strum specs ([perform.<cast>].strum) of the check's cast member and hand."""
    if not ctx.project:
        raise CheckError("strum: needs a project (its [perform.<cast>] strum spec and timeline)")
    cast = args.get("cast")
    if cast is None:
        if len(ctx.project.cast) != 1:
            raise CheckError("strum: say which cast member (cast = \"...\")")
        cast = ctx.project.cast[0]["name"]
    spec = (ctx.project.data.get("perform", {}).get(cast) or {}).get("strum")
    if not spec:
        raise CheckError(f"strum: [perform.{cast}] has no strum")
    specs = [spec] if isinstance(spec, dict) else list(spec)
    side = args.get("hand", "R")
    specs = [s for s in specs if s.get("hand", "R") == side]
    if not specs:
        raise CheckError(f"strum: [perform.{cast}] strums no {side} hand")
    return cast, specs


def _timeline(ctx, specs):
    path = ctx.project.path(specs[0].get("timeline", "audio/timeline.json"))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise CheckError(f"strum: the timeline {path} cannot be read: {e}") from None


def _seg_dist(P, A, B):
    """Distance of point P (3,) to each segment A[i]-B[i] (n, 3)."""
    ab = B - A
    t = np.clip(((P - A) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * ab), axis=1)


@metric
class Strum(Metric):
    name = "strum"
    doc = ("The pick against the strings at every stroke of a strum spec: the largest distance (mm) of the pick tip from the "
           "nearest string at the strike of a DOWN stroke (the pick meets the low E at its strike time), with the timing of "
           "the strikes in `detail` (when the tip really crosses the first string against the planned strike time). The "
           "strokes are the project's: `[perform.<cast>] strum` replayed on its timeline.")
    args = {"cast": "cast member whose strum is checked (default: the only one)", "hand": "R (default) | L: the pick hand",
            "pick": "the pick object (default <prop>_pick, built with the guitar)",
            "tip": "metres from the pick object's origin to its tip along its x (default: the card's pick.tip, 0.008)",
            "strokes": "down (default: the value is the worst down stroke) | all"}
    default_max = 10.0

    def needs(self, args, ctx, need):
        cast, specs = _specs(ctx, args)
        prop = specs[0].get("prop")
        if not prop:
            raise CheckError("strum: the strum spec names no `prop`")
        pick = args.get("pick", f"{prop}_pick")
        need.prop_use(prop)
        need.obj(prop, pick)
        return {"cast": cast, "specs": specs, "prop": prop, "pick": pick}

    def compute(self, args, ctx, data, frames, st):
        use = data.use(st["prop"])
        grips = {g["name"]: g for g in use.get("grip", [])}
        entry = grips.get(st["specs"][0].get("grip", "strum"))
        neck = next((g for g in grips.values() if g.get("type") == "neck"), None)
        if entry is None or neck is None:
            raise CheckError(f"strum: prop {st['prop']!r} needs use.grip entries `strum` and one of type neck")
        a, c, n = (np.asarray(entry[k], float) for k in ("along", "across", "normal"))
        centre = np.asarray(entry["center"], float)
        tip_off = float(args.get("tip", (entry.get("pick") or {}).get("tip", 0.008)))
        s_c = float(centre @ a)
        A = np.array([FR.string_point(neck, s, s_c) for s in range(6, 0, -1)])
        sigma = abs(float((A[-1] - A[0]) @ c)) / 2.0
        strings = [(np.asarray(g["nut"], float), np.asarray(g["bridge"], float)) for g in neck["strings"]]
        SA, SB = np.array([p[0] for p in strings]), np.array([p[1] for p in strings])
        frames = np.asarray(frames)
        ts = np.array([ctx.project.time(f) for f in frames])
        Mroot, Mpick = data.obj(st["prop"], frames), data.obj(st["pick"], frames)
        tip_w = np.einsum("fij,j->fi", Mpick[:, :3, :3], np.array([tip_off, 0.0, 0.0])) + Mpick[:, :3, 3]
        tip = np.einsum("fij,fj->fi", np.linalg.inv(Mroot)[:, :3, :3], tip_w) + np.linalg.inv(Mroot)[:, :3, 3]   # prop frame
        u = (tip - centre) @ c
        h = (tip - centre) @ n
        tl = _timeline(ctx, st["specs"])
        strokes = sorted((s for sp in st["specs"] for s in SM.plan(sp, tl, ctx.project.duration)), key=lambda s: s.t)
        rows = []
        for s in strokes:
            if not (ts[0] <= s.t <= ts[-1]):
                continue
            level = -s.dir * sigma
            near = np.where((ts >= s.t - 0.12) & (ts <= s.t + 0.12))[0]
            cross = None
            for k in near[:-1]:
                if (u[k] - level) * s.dir <= 0.0 < (u[k + 1] - level) * s.dir:        # moving the stroke's way through the first string
                    cross = ts[k] + (level - u[k]) / (u[k + 1] - u[k]) * (ts[k + 1] - ts[k])
                    break
            t_ref = cross if cross is not None else s.t
            p = np.array([np.interp(t_ref, ts, tip[:, i]) for i in range(3)])
            d = float(_seg_dist(p, SA, SB).min()) * 1000.0
            rows.append({"t": round(s.t, 3), "dir": s.dir, "accent": s.accent, "dist_mm": round(d, 2),
                         "error_ms": None if cross is None else round((cross - s.t) * 1000.0, 1)})
        if not rows:
            raise CheckError("strum: no stroke of the spec falls inside the checked frames")
        want = [r for r in rows if r["dir"] > 0] if args.get("strokes", "down") == "down" else rows
        miss = [r for r in rows if r["error_ms"] is None]
        err = np.array([r["error_ms"] for r in rows if r["error_ms"] is not None], float)
        stats = {}
        if len(err):
            stats = {"mean_ms": round(float(err.mean()), 1), "median_ms": round(float(np.median(err)), 1),
                     "p95_abs_ms": round(float(np.percentile(np.abs(err), 95)), 1), "max_abs_ms": round(float(np.abs(err).max()), 1),
                     "std_ms": round(float(err.std()), 1)}
        vals = [(r["dist_mm"] if r["error_ms"] is not None else max(r["dist_mm"], 50.0)) for r in want]
        k = int(np.argmax(vals))
        worst = sorted(want, key=lambda r: -r["dist_mm"])[:5]
        return float(vals[k]), {"at_frame": int(round(ctx.project.frame(want[k]["t"]))), "strokes": len(rows),
                                "down": sum(1 for r in rows if r["dir"] > 0), "missed": len(miss),
                                "timing": stats, "sigma_mm": round(sigma * 1000.0, 1),
                                "tip_range_mm": {"u": [round(float(u.min()) * 1000, 1), round(float(u.max()) * 1000, 1)],
                                                 "h": [round(float(h.min()) * 1000, 1), round(float(h.max()) * 1000, 1)]},
                                "worst": worst, "missed_strokes": [r["t"] for r in miss][:8]}


TRUNK = ("upper_body", "upper_body2", "lower_body", "neck", "head", "leg.L", "leg.R", "knee.L", "knee.R")


@metric
class PropBody(Metric):
    name = "prop_body"
    doc = ("How deep a prop's geometry sinks into the character's collision bodies (rig.json: the torso, hips, legs, neck and "
           "head by default, not the arms and hands that hold it): the deepest vertex over the frames (mm). The bodies are "
           "capsules a little fatter than the skin, so a few millimetres is a prop resting on the body; look before fixing.")
    args = {"cast": "cast member (needs its rig.json)", "prop": "the prop (its evaluated geometry, as a render shows it)",
            "bones": f"semantic bone names whose bodies count (default {', '.join(TRUNK)})",
            "exclude": "object names or patterns of the prop that do not count (a cable, a pick)",
            "samples": "vertices tested (default 4000, evenly spread)", "frame": "frame the prop's geometry is read at"}
    default_max = 8.0

    def needs(self, args, ctx, need):
        arm, rig = ctx.cast(args)
        if rig is None:
            raise CheckError("prop_body: needs the model's rig.json (cast with rig/asset, or rig = path)")
        if not args.get("prop"):
            raise CheckError("prop_body: give `prop`")
        wanted = []
        for sem in _list(args.get("bones")) or TRUNK:
            name = rig["map"].get(sem, sem)
            if name in rig["bones"]:
                wanted.append(name)
        shapes = geom.Shapes()
        used = geom.model_shapes(shapes, dict(rig, bodies=[b for b in rig["bodies"] if b.get("bone") in wanted]), arm)
        if not used:
            raise CheckError(f"prop_body: the model has no collision bodies on {wanted}")
        need.bone(arm, *sorted({b["bone"] for b in used.values()}))
        frame = args.get("frame")
        mi = need.mesh(prop=args["prop"], exclude=_list(args.get("exclude")), frame=frame)
        need.obj(args["prop"])
        return {"arm": arm, "shapes": shapes, "mesh": mi, "rig": rig, "bones": [b["bone"] for b in used.values()]}

    def compute(self, args, ctx, data, frames, st):
        m = data.mesh(st["mesh"])
        root = m["roots"][args["prop"]]
        V = m["vertices"]
        n = int(args.get("samples", 4000))
        V = V[np.linspace(0, len(V) - 1, min(n, len(V))).astype(int)]
        X0 = (V - root[:3, 3]) @ np.linalg.inv(root[:3, :3]).T                            # the prop's own frame
        frames = np.asarray(frames)
        Mroot = data.obj(args["prop"], frames)
        P = st["shapes"].pack()
        labels = [f"body:{st['rig']['bodies'][mm]['bone']}" if mm is not None else "?" for kind in geom.KINDS
                  for mm in P[kind]["model"]]
        Rs, ps, _R0, _p0 = data.source_frames(st["shapes"].sources, frames)
        r0 = np.zeros(len(X0))
        best = np.zeros(len(frames))
        where = []
        for k in range(len(frames)):
            X = np.einsum("ij,mj->mi", Mroot[k][:3, :3], X0) + Mroot[k][:3, 3]
            res = geom.penetration(X, r0, geom.world(P, Rs[k], ps[k]), None)
            rows = []
            for kind in geom.KINDS:
                if kind in res:
                    rows.append(res[kind][0])
            S = np.concatenate(rows, 1)
            j = np.unravel_index(int(np.argmax(S)), S.shape)
            best[k] = S[j]
            where.append((int(j[0]), labels[j[1]]))
        k = int(np.argmax(best))
        mm = best * 1000.0
        return float(max(mm[k], 0.0)), {"at_frame": int(frames[k]), "into": where[k][1], "vertex": where[k][0],
                                        "p95_mm": round(float(np.percentile(np.maximum(mm, 0), 95)), 2),
                                        "frames_over_3mm": int((mm > 3.0).sum()), "vertices": int(len(X0)),
                                        "bodies": sorted(set(st["bones"]))}
