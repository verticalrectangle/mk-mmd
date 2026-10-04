"""Timing curves of the cut effects (docs/design.md: Transitions and inserts), stdlib only: easing, geometric zooms and the
springy pop of an insert. Every curve maps progress 0..1 to a value that is exactly 0 at 0 and 1 at 1 (a pop overshoots
in between)."""
import math

EASES = ("in", "out", "inout")


def clamp01(u):
    return 0.0 if u <= 0.0 else 1.0 if u >= 1.0 else float(u)


def ease(u, kind="in"):
    """Progress -> eased progress. `in` starts slowly and ends fast (u^2), `out` the reverse, `inout` is smoothstep."""
    u = clamp01(u)
    if kind == "in":
        return u * u
    if kind == "out":
        return 1.0 - (1.0 - u) ** 2
    if kind == "inout":
        return u * u * (3.0 - 2.0 * u)
    raise ValueError(f"ease {kind!r}: expected one of {EASES}")


def geometric(lo, hi, p):
    """Interpolate a zoom factor between `lo` and `hi`: equal steps of p are equal ratios (a steady apparent speed)."""
    return float(lo) * (float(hi) / float(lo)) ** p


def lerp(a, b, p):
    return a + (b - a) * p


# ---------------------------------------------------------------- the springy pop
def _overshoot_of(c):
    """Peak excess over 1 of the ease-out-back curve with coefficient c: 4 c^3 / (27 (c + 1)^2)."""
    return 4.0 * c ** 3 / (27.0 * (c + 1.0) ** 2)


def back_coeff(overshoot):
    """The ease-out-back coefficient whose curve peaks `overshoot` (0.1 = 10 %) above its end value."""
    if overshoot <= 0.0:
        return 0.0
    lo, hi = 0.0, 1.0
    while _overshoot_of(hi) < overshoot:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError(f"overshoot {overshoot} is out of reach")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _overshoot_of(mid) < overshoot:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pop(u, overshoot=0.0):
    """0 -> 1 with a cubic ease-out that overshoots by `overshoot` (a fraction of the end value) before settling."""
    u = clamp01(u)
    c = back_coeff(overshoot)
    return 1.0 + (c + 1.0) * (u - 1.0) ** 3 + c * (u - 1.0) ** 2


def stagger(u, start, span):
    """The progress of a part that starts at `start` and takes `span` of the whole (all in 0..1 of the parent's time)."""
    return clamp01((u - start) / span) if span > 0 else (1.0 if u >= start else 0.0)
