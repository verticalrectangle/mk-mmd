"""JSON conversion for numpy, mathutils and pathlib values (shared by the CLI and the Blender side)."""
import json
import math
from pathlib import PurePath


def to_jsonable(x, precision=None):
    """Plain JSON types from numpy arrays/scalars, mathutils Vector/Quaternion/Matrix/Euler, paths, sets and tuples.
    Floats are rounded to `precision` decimals when given; NaN/inf become None."""
    if x is None or isinstance(x, (bool, str, int)):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, precision) if precision is not None else x
    if isinstance(x, PurePath):
        return str(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v, precision) for k, v in x.items()}
    if hasattr(x, "tolist") and not isinstance(x, (list, tuple)):   # numpy array or scalar
        return to_jsonable(x.tolist(), precision)
    if isinstance(x, (list, tuple, set, frozenset)):
        return [to_jsonable(v, precision) for v in x]
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    if hasattr(x, "__len__") and hasattr(x, "__getitem__"):          # mathutils types
        return [to_jsonable(x[i], precision) for i in range(len(x))]
    if hasattr(x, "__float__"):
        return to_jsonable(float(x), precision)
    return str(x)


def dumps(obj, precision=None, indent=1) -> str:
    return json.dumps(to_jsonable(obj, precision), ensure_ascii=False, indent=indent)
