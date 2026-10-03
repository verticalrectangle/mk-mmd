"""Frame specs for --frames (see docs/design.md: Conventions)."""


class FrameSpecError(ValueError):
    pass


def parse(spec, fps=None, frame0=None, duration=None):
    """Parse a frame spec into a list of frame numbers (ints, or floats for sub-frames).

    `181:280` inclusive range, `181:280:5` with a step, `100,140,200` a list, `t=1.5:3.0[:0.5]` clip seconds (needs a
    project), `all` the project's clip. Specs can be joined with commas: `100,200:210`."""
    spec = str(spec).strip()
    if not spec:
        raise FrameSpecError("empty frame spec")
    if spec == "all":
        _need_project(fps, frame0, duration, spec)
        n = int(round(duration * fps))
        return list(range(int(frame0), int(frame0) + n))
    if spec.startswith("t="):
        _need_project(fps, frame0, None, spec)
        parts = [float(p) for p in spec[2:].split(":")]
        if len(parts) == 1:
            return [_num(frame0 + parts[0] * fps)]
        t0, t1 = parts[0], parts[1]
        step = parts[2] if len(parts) > 2 else 1.0 / fps
        if step <= 0 or t1 < t0:
            raise FrameSpecError(f"bad time range {spec!r}")
        out, k = [], 0
        while t0 + k * step <= t1 + 1e-9:
            out.append(_num(frame0 + (t0 + k * step) * fps))
            k += 1
        return out
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = part.split(":")
            if len(bits) not in (2, 3):
                raise FrameSpecError(f"bad range {part!r}")
            a, b = int(bits[0]), int(bits[1])
            step = int(bits[2]) if len(bits) == 3 else 1
            if step <= 0 or b < a:
                raise FrameSpecError(f"bad range {part!r}")
            out.extend(range(a, b + 1, step))
        else:
            out.append(_num(float(part)))
    if not out:
        raise FrameSpecError(f"no frames in {spec!r}")
    return out


def _num(x):
    return int(round(x)) if abs(x - round(x)) < 1e-6 else float(x)


def _need_project(fps, frame0, duration, spec):
    if fps is None or frame0 is None or (duration is None and spec == "all"):
        raise FrameSpecError(f"{spec!r} needs a project (mk.toml with fps, frame0, duration)")
