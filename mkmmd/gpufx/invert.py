"""invert on the GPU (mkr-post fx/invert.rs): EraFx._invert_level (eased, on its thinned hits, times the strength), its
hue turn and its region; the inversion, the hue turn (OpenCV's float HSV) and the mix drawn by mkr-post."""
from .fields import f32, region


def ops(ffx, fx, frame):
    v = ffx._invert_level(fx, frame)
    if v <= 1e-4:
        return []
    reg = region(ffx, fx, frame)
    if reg is False:
        return None
    hue = fx["hue"]
    return [{"op": "invert", "hue": f32(hue) if abs(hue) % 360.0 > 1e-6 else None, "v": f32(v), "mask": reg}]
