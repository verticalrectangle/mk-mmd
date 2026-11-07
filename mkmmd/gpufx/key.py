"""key on the GPU (mkr-post fx/key.rs): the key's tolerance on the frame and on each frame of its `burn` trail (the same
FrameFx.key_tolerance the CPU path uses: its keys, a `follow` stem's loudness, a pump on the hits), the centre its
`spread` opens from (mkr-post takes the spread off each pixel's tolerance, as KeyFx.key_spread does), the planes' splits,
the constants of the distance and the lay as mkmmd.colorkey works them in float32, and what shows through: a flat
colour, the cut's frame `delay` before (a PNG mkr-post reads), the picture itself where the cut has no such frame, or a
clip's frame decoded and fitted here as the footage kind's is (mkmmd.gpufx.footage.hand). mkr-post draws the distance,
the mattes (kept for the trail as bytes, as the CPU path keeps them), the lay, the mask and the mix."""
import numpy as np

from .. import colorkey as CKF
from .. import framefx as FF
from ..core import framefx as FX
from .numbers import f32, f32s

SPACES = {"rgb": 0, "chroma": 1, "luma": 2}                 # the codes fx_key.wgsl measures the distance by


def _tolerance(ffx, fx, g):
    """The tolerance a key keys frame `g` at before its spread: (float32 of KeyFx.key_tolerance, the spread's centre px
    there), as KeyFx.key_spread reads them (the centre on the span's first frame before the span)."""
    centre = [0.0, 0.0]
    if fx["spread"]:
        (cx, cy), _ = ffx._at(fx, max(g, fx["f0"]))
        centre = [f32(cx), f32(cy)]
    return f32(ffx.key_tolerance(fx, g)), centre


def ops(ffx, fx, frame):
    paths = ffx.paths
    if paths is None or paths.scratch is None:
        return None
    t = ffx._t(frame)
    if FX.strength(fx, t) <= 0.0:
        return []
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
    tol, centre = _tolerance(ffx, fx, frame)
    mask = None
    if fx["mask"] is not None:
        fig = paths.figure(FX.figure_key(fx), frame)
        if not paths.fits(fig):
            return None
        mask = {"figure": str(fig), "inside": fx["mask"] == "figure"}
    trail = []
    for j in range(1, fx["burn"] + 1):                      # the cut's frames before, keyed at their own tolerance
        g = frame - j
        p = paths.frame(g) if g >= ffx.frame0 else None
        if p is None:
            continue
        if not paths.fits(p):
            return None
        tg, cg = _tolerance(ffx, fx, g)
        trail.append({"path": str(p), "tol": tg, "centre": cg, "factor": f32(fx["fade"] ** j / 255.0)})
    back, clip = {"source": "picture"}, None
    if fx["source"] == "back":
        back = {"source": "colour", "color": f32s(fx["back"])}
    elif fx["source"] == "delay":
        p = paths.frame(max(frame - fx["delay"], ffx.frame0))
        if p is not None:
            if not paths.fits(p):
                return None
            back = {"source": "png", "path": str(p)}
    else:
        if ffx._footage_of is None:
            return None
        c = FF.clip_of(fx, ffx._footage_of)
        i = c.index((frame - fx["f0"]) / ffx.fps, fx["in"], fx["rate"], fx["loop"])
        clip = {"name": fx["clip"], "index": int(i), "fit": "cover", "focus": [float(v) for v in fx["focus"]],
                "mirror": bool(fx["mirror"])}
        back = {"source": "clip"}
    col = np.asarray(fx["color"], np.float32)
    yc = np.float32(col @ CKF.LUMA)
    kb, kr = np.float32((col[2] - yc) / CKF.CB), np.float32((col[0] - yc) / CKF.CR)
    ref = [f32(yc), f32(kb), f32(kr), f32(1.0 / CKF.CB), f32(1.0 / CKF.CR)]
    spill = fx["spill"]
    op = {"op": "key", "space": SPACES[fx["space"]], "color": f32s(col), "ref": ref, "tol": tol, "centre": centre,
          "split": f32s(fx["split"]), "soft": f32(fx["soft"]), "spread": f32(fx["spread"] / ffx.size[1]),
          "inv": f32(1.0 / fx["soft"]) if fx["soft"] > 0 else None, "bands": int(fx["bands"]),
          "bscale": f32(1.0 / fx["bands"]) if fx["bands"] else 0.0, "gain": f32s(fx["gain"]), "back": back,
          "clip": clip, "trail": trail,
          "spill": None if spill is None else {"color": f32s(spill["color"]), "amount": f32(spill["amount"]),
                                               "edge": f32(fx["soft"] + spill["reach"]),
                                               "inv": f32(1.0 / spill["reach"])},
          "mask": mask, "s": [f32(s), f32(1.0 - s)]}
    return [op]
