"""Numbers as the CPU path's numpy uses them, for mkr-post's ops: a value numpy casts to float32 is sent cast (JSON
carries it exactly, whatever float parser reads it), and a float64 bound a float32 array is compared with is sent as the
float32 the comparison comes to."""
import math

import numpy as np


def f32(v):
    """`v` as numpy casts it to float32 (a Python float in a float32 array's arithmetic), as a Python float."""
    return float(np.float32(v))


def f32s(vs):
    return [f32(v) for v in vs]


def below(x):
    """The float32 L with r < L exactly when r < x, for every float32 r (x a float64 bound, as numpy compares a float32
    array with a float64 scalar): x itself when it is a float32, else the float32 just above it."""
    d = np.float32(x)
    if float(d) > x:
        d = np.nextafter(d, np.float32(-math.inf))
    return float(d) if float(d) == x else float(np.nextafter(d, np.float32(math.inf)))
