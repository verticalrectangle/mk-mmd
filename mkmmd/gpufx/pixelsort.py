"""pixelsort on the GPU (mkr-post fx/pixelsort.rs): the strength from PictureFx.pixelsort_at, the key's quantisation as
mkmmd.picturefx.sort_runs packs it (bits by the frame's pixels), its mask's figure layer. A `rect` (postgpu) and lines
longer than a workgroup sorts are left to the CPU.

A pixel sort is chaotic in its input: a key a float's last bits higher can land in the next of its 16-bit steps, or a
pixel cross the run's threshold, and pixels move along their run (on Crimson, 1e-6 of noise in the picture it sorts
moved thousands of pixels by up to 99 levels). mkr-post's sort matches the CPU's within a level on the same picture,
but the effects before it drawn on the GPU (each within a level) would hand it another, so it is FIRST: on a frame it
sorts, the entries before it are drawn on the CPU (mkmmd.postgpu.first) and it sorts exactly the CPU path's picture."""
from ..core import framefx as FX
from .numbers import f32

FIRST = True                # the first op mkr-post draws on a frame it sorts (mkmmd.postgpu.first)
BY = {"luma": 0, "hue": 1, "sat": 2}
LINE = 4096                 # fx_pixelsort.wgsl's longest line
LANES = 8                   # OpenCV's float32 vector width here (AVX2): RGB2HSV works the rest of a row out one by one


def ops(ffx, fx, frame):
    paths = ffx.paths
    if paths is None:
        return None
    s = ffx.pixelsort_at(fx, frame)
    if s <= 1e-3:
        return []
    w, h = ffx.size
    if (w if fx["axis"] == "x" else h) > LINE:
        return None
    ib = max(int(w * h).bit_length(), 1)
    kb = min(16, 62 - 2 * ib)
    op = {"op": "pixelsort", "axis": 0 if fx["axis"] == "x" else 1, "by": BY[fx["by"]], "reverse": fx["reverse"],
          "top": (1 << kb) - 1, "tail": w - w % LANES, "lo": f32(fx["lo"]), "hi": f32(fx["hi"])}
    if fx["mask"] is not None:
        fig = paths.figure(FX.figure_key(fx), frame)
        if not paths.fits(fig):
            return None
        op["mask"] = {"figure": str(fig), "inside": fx["mask"] == "figure"}
    if s < 1.0:
        op["mix"] = s
    return [op]
