"""ribbon_clear: do the scene's [[ribbon]]s keep clear of the cast's bodies (docs/design.md: Checks, Ribbons)?

The ribbons stage keeps each ribbon's centre line on its root (`mk_ribbon`: x, y, z in the root's frame and the radius
that holds the surface at its fattest there, per point) and keys the root's `grow`, `tail` and `thick`. The metric
samples those and the root's world matrix on every frame, takes the part the ribbon shows there (mkmmd.core.ribbon
.visible), as fat as its pulse makes it, and measures it against the characters' collision bodies (rig.json, as the
scene poses them: after the sim, the hair and skirt bodies where they swung): exact for a round ribbon, and for a flat
one as if it were as thick as it is wide."""
from fnmatch import fnmatchcase

import numpy as np

from ..core import ribbon as RB
from ..solvers import geom
from . import CheckError, Metric, metric


def _list(v):
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


@metric
class RibbonClear(Metric):
    name = "ribbon_clear"
    doc = ("How close the scene's [[ribbon]]s come to the characters: on every frame the part of each ribbon that shows "
           "(its centre line and the radius round it that holds its surface, pulses included) against the collision "
           "bodies of rig.json as the scene poses them. Value: the least gap (m) over frames, ribbons and bodies "
           "(negative: into a body); detail: which ribbon, on which frame, nearest which body.")
    args = {"ribbons": "ribbon names or fnmatch patterns (default every [[ribbon]] of the project)",
            "cast": "cast member or list of them (default every member with a rig.json)"}
    default_min = 0.0

    def needs(self, args, ctx, need):
        if ctx.project is None:
            raise CheckError("ribbon_clear: needs the project (its [[ribbon]] entries)")
        entries = ctx.project.data.get("ribbon") or []
        entries = [entries] if isinstance(entries, dict) else entries
        have = [e.get("name") for e in entries if isinstance(e, dict) and e.get("name")]
        pats = _list(args.get("ribbons")) or have
        names = [n for n in have if any(fnmatchcase(n, p) for p in pats)]
        if not names:
            raise CheckError(f"ribbon_clear: no ribbon matches {pats} (the project's: {have or 'none'})")
        members = _list(args.get("cast")) or [c.get("name") for c in ctx.project.cast]
        cast = []
        for c in members:
            arm, rig = ctx.cast({"cast": c})
            if rig is None or not rig.get("bodies"):
                if args.get("cast"):
                    raise CheckError(f"ribbon_clear: cast member {c!r} has no collision bodies (rig.json)")
                continue
            need.bone(arm, *[b["bone"] for b in rig["bodies"] if b.get("bone") and "geom" in b])
            cast.append((c, arm, rig))
        if not cast:
            raise CheckError("ribbon_clear: no cast member with collision bodies (rig.json)")
        need.obj(*names)
        rows = {n: (need.expr(f'list(bpy.data.objects[{n!r}]["mk_ribbon"])'),
                    need.expr(f'[bpy.data.objects[{n!r}][k] for k in ("grow", "tail", "thick")]')) for n in names}
        return {"names": names, "cast": cast, "rows": rows}

    def compute(self, args, ctx, data, frames, st):
        shapes = geom.Shapes()
        owner = {}
        for c, arm, rig in st["cast"]:
            geom.model_shapes(shapes, rig, arm)
            owner[arm] = c
        P = shapes.pack()
        labels = [f"{owner.get(shapes.sources[j][1], shapes.sources[j][1])}:{shapes.sources[j][2]}"
                  for kind in geom.KINDS for j in P[kind]["src"]]
        Rs, ps, _, _ = data.source_frames(shapes.sources, frames)
        frames = np.asarray(frames)
        best, where, per = np.inf, None, {}
        for name in st["names"]:
            i_pts, i_state = st["rows"][name]
            pts = np.asarray(data.expr(i_pts, frames)[0], float).reshape(-1, 4)
            X0, rad = pts[:, :3], pts[:, 3]
            s = RB.arclen(X0)
            state = np.asarray(data.expr(i_state, frames), float).reshape(len(frames), 3)
            M = data.obj(name, frames)
            low, at = np.inf, None
            for k in range(len(frames)):
                vis = np.flatnonzero(RB.visible(s, s[-1], state[k, 0], state[k, 1]))
                if not len(vis):
                    continue
                X = X0[vis] @ M[k][:3, :3].T + M[k][:3, 3]
                scale = float(np.cbrt(abs(np.linalg.det(M[k][:3, :3]))))
                res = geom.penetration(X, rad[vis] * state[k, 2] * scale, geom.world(P, Rs[k], ps[k]), None)
                base = 0
                for kind in geom.KINDS:
                    if kind not in res:
                        continue
                    pen = res[kind][0]
                    j = np.unravel_index(int(np.argmax(pen)), pen.shape)
                    if -pen[j] < low:
                        low, at = float(-pen[j]), (int(frames[k]), labels[base + j[1]], round(float(s[vis[j[0]]]), 3))
                    base += pen.shape[1]
            if at is None:
                continue
            per[name] = {"gap": round(low, 4), "at_frame": at[0], "body": at[1], "at_m": at[2]}
            if low < best:
                best, where = low, (name, *at)
        if where is None:
            raise CheckError(f"ribbon_clear: none of {st['names']} shows on the frames checked")
        return round(best, 4), {"ribbon": where[0], "at_frame": where[1], "body": where[2], "at_m": where[3],
                                "ribbons": per}
