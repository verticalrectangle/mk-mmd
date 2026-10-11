"""Metrics about the camera: framing (per output aspect), occlusion and camera_inside (ray tests in Blender)."""
import numpy as np

from .. import bridge
from . import CheckError, Metric, is_expr, metric


def project(points, cam, k, size):
    """Normalised image coordinates (n, 2) (0..1, origin bottom-left) and depth (n,) of world points for camera
    sample k rendered at size (w, h). Follows Blender: sensor fit AUTO puts the sensor width on the larger image
    side; the lens shift (fractions of the larger side) moves the VIEW WINDOW, so the picture moves the other way: a
    positive shift_x moves it left, a positive shift_y down (probed against world_to_camera_view in Blender 4.2.3)."""
    w, h = size
    M = cam["world"][k]
    R, t = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0, keepdims=True), M[:3, 3]
    pc = (points - t) @ R                                   # camera space: x right, y up, -z forward
    depth = -pc[:, 2]
    fit = cam["sensor_fit"].get(cam["names"][k], "AUTO")
    sw, sh = cam["sensor"][k]
    lens = cam["lens"][k]
    if fit == "VERTICAL":
        fpix = lens / sh * h
    elif fit == "HORIZONTAL":
        fpix = lens / sw * w
    else:
        fpix = lens / sw * max(w, h)
    big = max(w, h)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (w / 2 + fpix * pc[:, 0] / depth - cam["shift"][k][0] * big) / w
        v = (h / 2 + fpix * pc[:, 1] / depth - cam["shift"][k][1] * big) / h
    return np.stack([u, v], 1), depth


def _point_list(args, default):
    pts = args.get("points") or default
    return [pts] if isinstance(pts, str) else list(pts)


@metric
class Framing(Metric):
    name = "framing"
    doc = ("Subject inside the frame for every output aspect: each point projected through the active camera "
           "(timeline markers respected) at each output's size. Value: the smallest margin to the safe area over all "
           "frames, points and outputs, as a fraction of the frame (negative = outside the safe area).")
    args = {"points": "bones (head position) or expressions (default head and neck)",
            "cast": "cast member whose bones are meant", "outputs": "output names (default: all in mk.toml)",
            "safe": "safe-area margin as a fraction of each side (default 0.05)"}

    def needs(self, args, ctx, need):
        arm, _ = ctx.cast(args)
        pts = _point_list(args, ["head", "neck"])
        st = {"arm": arm, "bones": [], "exprs": []}
        for p in pts:
            if is_expr(p):
                st["exprs"].append((p, need.expr(p)))
            else:
                need.bone(arm, p)
                st["bones"].append(p)
        need.camera = True
        return st

    def compute(self, args, ctx, data, frames, st):
        outs = args.get("outputs")
        if ctx.project and ctx.project.outputs:
            sizes = {o.name: o.size for o in ctx.project.outputs if not outs or o.name in outs}
        else:
            r = data.meta["camera"]["resolution"]
            sizes = {"scene": tuple(r)}
        if not sizes:
            raise CheckError(f"framing: no outputs match {outs}")
        safe = float(args.get("safe", 0.05))
        cols = []
        labels = []
        if st["bones"]:
            W, _, names = data.bones(st["arm"], st["bones"], frames)
            cols.append(W[..., :3, 3])
            labels += names
        for p, i in st["exprs"]:
            v = np.asarray(data.expr(i, frames), float).reshape(len(frames), -1, 3)
            cols.append(v)
            labels += [p] * v.shape[1]
        P = np.concatenate(cols, 1)
        cam = data.camera(frames)
        worst, detail, at = np.inf, {}, None
        for name, size in sizes.items():
            margins = np.zeros(len(frames))
            where = []
            for k in range(len(frames)):
                if cam["names"][k] is None:
                    raise CheckError("framing: the scene has no active camera")
                uv, depth = project(P[k], cam, k, size)
                m = np.minimum.reduce([uv[:, 0], 1 - uv[:, 0], uv[:, 1], 1 - uv[:, 1]]) - safe
                m = np.where(depth > 0, m, -1.0)
                j = int(np.argmin(m))
                margins[k] = m[j]
                where.append(labels[j])
            k = int(np.argmin(margins))
            bad = np.asarray(frames)[margins < 0]
            detail[name] = {"min_margin": round(float(margins[k]), 4), "at_frame": int(frames[k]),
                            "point": where[k], "camera": cam["names"][k], "frames_outside": int(len(bad)),
                            "first_outside": [int(f) for f in bad[:10]]}
            if float(margins[k]) < worst:
                worst, at = float(margins[k]), int(frames[k])
        detail["at_frame"] = at
        return worst, detail


@metric
class Occlusion(Metric):
    name = "occlusion"
    doc = ("Is the subject hidden behind something? Rays from the active camera to subject points; hits on the "
           "subject's own meshes do not count. Value: the largest share of blocked points on any frame (0..1).")
    args = {"points": "expressions giving a point or a list of points (default the cast's head and chest)",
            "cast": "cast member: its meshes are the subject (or ignore_models = [armature names])",
            "ignore": "object names that never count as blockers (glass, hair cards)"}
    sampled = False

    def needs(self, args, ctx, need):
        arm, _ = ctx.cast(args)
        return {"arm": arm}

    def compute(self, args, ctx, data, frames, st):
        arm = st["arm"]
        pts = _point_list(args, [f'bone("head", {arm!r}).head' if arm else 'bone("head").head',
                                 f'bone("upper_body2", {arm!r}).head' if arm else 'bone("upper_body").head'])
        rows = bridge.run("visibility", {"frames": frames, "points": pts, "ignore": args.get("ignore", []),
                                         "ignore_models": args.get("ignore_models", [arm]), "armature": arm or None},
                          blend=ctx.scene, project=ctx.project, timeout=3600)
        share = np.array([r["blocked"] / max(r["points"], 1) for r in rows])
        k = int(np.argmax(share))
        by = {}
        for r in rows:
            for n, c in r["by"].items():
                by[n] = by.get(n, 0) + c
        return float(share[k]), {"at_frame": int(frames[k]), "frames_blocked": int((share > 0).sum()),
                                 "blockers": dict(sorted(by.items(), key=lambda kv: -kv[1])[:5])}


@metric
class CameraInside(Metric):
    name = "camera_inside"
    doc = ("Camera inside a closed mesh a render shows (a wall, a car body, a head): rays in six directions all hit "
           "back faces of the same object; hidden objects (colliders) and `ignore` are passed through. Value: the "
           "number of such frames.")
    args = {"ignore": "object names a camera may sit inside (glow and haze volumes)"}
    sampled = False

    def compute(self, args, ctx, data, frames, st):
        rows = bridge.run("visibility", {"frames": frames, "points": [], "ignore": args.get("ignore", [])},
                          blend=ctx.scene, project=ctx.project, timeout=3600)
        bad = [r for r in rows if r["inside"]]
        return float(len(bad)), {"at_frame": bad[0]["frame"] if bad else None, "frames": [r["frame"] for r in bad[:20]],
                                 "inside": sorted({r["inside"] for r in bad})}
