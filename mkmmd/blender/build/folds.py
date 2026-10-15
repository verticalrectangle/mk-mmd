"""folds: the fold shapes a joint angle drives (rig.json `folds`, mkmmd.core.folds), keyed on every frame of the build.

[folds.<cast>] keys (all optional: a member whose rig has folds gets them without a table)
  enabled = true    false: no fold shapes for this member (they stay at 0)
  depth = 1.0       scales every fold's weight
A member whose rig has no folds is left alone, and so is a fold on a piece the cloth stage simulates for the member
(its `piece` a cloth copy's name: the cloth makes its own folds there). The stage runs after sim, when every bone that
drives a fold has its final keys; the driving bones' rotations are read in the armature's own space (a model spinning in
the world turns no fold), and the weights are keyed LINEAR on the morph sliders (the bound placeholder, or the
meshes)."""
import bpy
import numpy as np

from ...core import folds as FD
from ...solvers import geom
from .. import keys as K
from .. import scene as S
from . import cloth as CL


def _mat(m):
    return np.array([list(r) for r in m], float)


def sample_rotations(arm, bones, frames):
    """{bone: (F, 3, 3)} rotation deltas of the bones in the armature's space (posed rotation times the rest one's
    inverse) on every frame."""
    sc = bpy.context.scene
    rest = {b: geom.unscaled(_mat(arm.data.bones[b].matrix_local))[:3, :3] for b in bones}
    out = {b: np.zeros((len(frames), 3, 3)) for b in bones}
    for i, f in enumerate(frames):
        sc.frame_set(int(f))
        ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
        for b in bones:
            out[b][i] = geom.unscaled(_mat(ev.pose.bones[b].matrix))[:3, :3] @ rest[b].T
    return out


def run(ctx):
    ctx.check_tables("folds")
    out = {}
    for name, m in ctx.cast.items():
        folds = list((m.rig or {}).get("folds") or [])
        spec = ctx.section("folds", name) or {}
        if not folds:
            continue
        if not spec.get("enabled", True):
            out[name] = {"skipped": "enabled = false"}
            continue
        cloth = set(CL.simulated(ctx, name, m))
        left = sorted({f["morph"] for f in folds if f.get("piece") in cloth})
        folds = [f for f in folds if f.get("piece") not in cloth]
        if not folds:
            out[name] = {"left_to_cloth": left}
            continue
        bones = sorted({f["driver"][k] for f in folds for k in ("bone", "ref")})
        missing = [b for b in bones if b not in m.arm.data.bones]
        if missing:
            ctx.log("WARNING", f"folds.{name}: the rig's fold drivers name bones the armature lacks: {missing}")
            continue
        frames = [int(f) for f in ctx.frames]
        D = sample_rotations(m.arm, bones, frames)
        dirs = {b: np.asarray(m.rig["bones"][b]["tail"], float) - np.asarray(m.rig["bones"][b]["head"], float)
                for b in bones}
        W = FD.weights(folds, D, dirs)
        depth = float(spec.get("depth", 1.0))
        holders = S.morph_holders(m.arm)
        keyed = {}
        for morph, w in W.items():
            vals = np.clip(np.asarray(w, float) * depth, 0.0, 1.0)
            if K.key_morph(holders, morph, frames, [float(v) for v in vals], interp="LINEAR"):
                keyed[morph] = round(float(vals.max()), 3)
        out[name] = {"folds": len(folds), "keyed": keyed, **({"left_to_cloth": left} if left else {})}
        ctx.log("folds", name, f"{len(keyed)} shapes keyed, most of each: {keyed}")
    return out
