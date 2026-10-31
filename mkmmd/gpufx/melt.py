"""melt on the GPU (mkr-post fx/melt.rs): the mode's numbers and tables from RemapFx.melt_at and the pieces of
mkmmd.core.remap.melt_field (melt_wave, ripple_amp, drip_columns, spread_taps, noise), as melt_field uses them, and
its mask (gpufx.masked)."""
import math

import numpy as np

from ..core import remap as RM
from .masked import masking
from .numbers import f32

MODES = {"breathe": 0, "swirl": 1, "ripple": 2, "drip": 3, "noise": 4}
TAU = 2.0 * np.pi


def ops(ffx, fx, frame):
    got = field(ffx, fx, frame)
    if got and fx["mask"] is not None:              # its figure is read only when the melt moves something
        if ffx.paths is None:
            return None
        mask = masking(ffx, fx, frame)
        if mask is False:
            return None
        got[0]["mask"] = mask
    return got


def field(ffx, fx, frame):
    """The melt's op without its mask: [] when nothing moves (its level 0)."""
    m = ffx.melt_at(fx, frame)
    mode, centre, level, scale, speed, age, unit = (m[k] for k in ("mode", "centre", "level", "scale", "speed",
                                                                    "age", "unit"))
    w, h = ffx.size
    op = {"op": "melt", "mode": MODES[mode], "c": [f32(centre[0]), f32(centre[1])], "k0": 0.0}
    if mode == "ripple":
        if m["front"]:
            rs, amp = RM.ripple_amp(level, age, m["front"], RM.corner_reach((w, h), centre))
            if not amp.any():
                return []
            op.update(n=len(rs), step=f32(rs[1] - rs[0]), table=(amp * np.float32(unit)).tolist())
        else:
            a = np.float32(level(age) * unit)
            if a == 0:
                return []
            op["k0"] = float(a)
        op.update(su=f32(scale * unit), tau=f32(TAU), phase=f32(speed * age))
        return [op]
    a = float(level(age))
    if a == 0.0:
        return []
    wave = RM.melt_wave(mode, speed, age)
    if mode == "breathe":
        g = max(a * wave, -0.9)
        op.update(k0=f32(g / (1.0 + g)), su=f32(scale * unit))
    elif mode == "swirl":
        op.update(k0=f32(math.radians(a * wave)), su=f32(scale * unit))
    elif mode == "drip":
        drips, col = RM.drip_columns(w, scale * unit, speed, age, m["seed"])
        op.update(k0=f32(a * unit), tau=f32(TAU), phase=f32(speed * age),
                  table=np.stack([drips, col], 1).astype(np.float32).ravel().tolist())
    else:
        cell = max(scale * unit, 2.0)
        ry, wy, my = RM.spread_taps(h, cell)
        rx, wx, mx = RM.spread_taps(w, cell)
        lat = RM.noise(m["seed"], (my, mx, 2), speed * age).astype(np.float32)
        rows = np.concatenate([ry[:, None].astype(np.float32), wy], 1)
        cols = np.concatenate([rx[:, None].astype(np.float32), wx], 1)
        op.update(k0=f32(a * unit), spread=f32(RM.SPREAD), n=int(mx),
                  table=np.concatenate([rows.ravel(), cols.ravel(), lat.ravel()]).tolist())
    return [op]
