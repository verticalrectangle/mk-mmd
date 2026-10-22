"""clip: where each shot camera starts seeing (docs/design.md: Shots: Clip ranges). On every frame the shots stage keyed
a camera over, its clip_start is keyed (CONSTANT) to mkmmd.core.clip.near of the nearest geometry it sees on that frame,
so the depth a render keeps is fine enough to hide what MMD models hide a hair behind the skin (drawn expressions).
Runs after everything that moves geometry, on the world a render sees: a freeze's held frame with the camera on its own
(mkmmd.blender.freeze)."""
import json

import bpy
import numpy as np

from ...core import clip as CL
from .. import depth as DP
from .. import freeze as FRZ
from .. import keys as K


def run(ctx):
    sc = bpy.context.scene
    table = json.loads(sc.get("mk_shots", "[]"))
    sizes = {o["name"]: o["size"] for o in ctx.project.get("outputs") or []}
    live = {}                                                # frame -> [(camera, picture size)]
    for e in table:
        a, b = e["keyed"]
        for out, cam in e["cameras"].items():
            size = sizes.get(out) or [sc.render.resolution_x, sc.render.resolution_y]
            for f in range(int(a), int(b) + 1):
                live.setdefault(f, []).append((cam, size))
    if not live:
        return {}
    depth, wins, got = DP.Depth(), FRZ.windows(sc), {}
    try:
        for f in sorted(live):
            FRZ.frame_set(sc, f, wins)
            depth.frame()
            for cam, size in live[f]:
                near = depth.near(bpy.data.objects[cam], size)
                if near is not None:
                    got.setdefault(cam, []).append((f, near))
    finally:
        FRZ.release()
    for cam, rows in got.items():
        K.set_fcurve(bpy.data.objects[cam].data, "clip_start", 0, [f for f, _ in rows], [v for _, v in rows],
                     interp="CONSTANT")
    v = np.array([n for rows in got.values() for _, n in rows])
    if not len(v):
        return {}
    return {"cameras": len(got), "frames": int(len(v)),
            "near_m": [round(float(v.min()), 4), round(float(np.median(v)), 3), round(float(v.max()), 2)],
            "touching": int((v <= CL.NEAR_MIN).sum())}
