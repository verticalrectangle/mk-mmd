"""Static instances of variable fonts, for the text stage (Blender reads a variable font's default instance only).

    python -m mkmmd.fontinst SRC.ttf OUT.ttf wght=420 opsz=72 SOFT=100 WONK=1
    python -m mkmmd.fontinst SRC.ttf --axes          # list the axes: tag, minimum, default, maximum

Pins every named axis at its value (limits are checked against the font's axes) and every other axis at its default,
then writes a plain static font whose family name carries the axes, so registered instances do not clash:
`mk assets add OUT.ttf --kind font --slug fraunces_italic` (docs/design.md: Text, fonts). CLI side: needs fontTools,
which `pip install fonttools` provides (Pillow-free, no Blender)."""
import os
import sys


def axes_of(path):
    """[(tag, minimum, default, maximum)] of a variable font's axes."""
    from fontTools.ttLib import TTFont
    font = TTFont(path)
    if "fvar" not in font:
        raise ValueError(f"{os.path.basename(path)} is not a variable font")
    return [(a.axisTag, a.minValue, a.defaultValue, a.maxValue) for a in font["fvar"].axes]


def instantiate(src, out, axes):
    """Write the static instance of `src` at `axes` ({tag: value}) to `out`; returns {tag: value} for every axis."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer
    have = {t: (lo, d, hi) for t, lo, d, hi in axes_of(src)}
    unknown = sorted(set(axes) - set(have))
    if unknown:
        raise ValueError(f"{os.path.basename(src)} has no axes {unknown} (it has {sorted(have)})")
    for tag, v in axes.items():
        lo, _, hi = have[tag]
        if not lo <= v <= hi:
            raise ValueError(f"axis {tag} = {v} is outside the font's range {lo}..{hi}")
    loc = {tag: float(axes.get(tag, d)) for tag, (lo, d, hi) in have.items()}
    font = instancer.instantiateVariableFont(TTFont(src), loc, inplace=False)
    suffix = "".join(f"{t}{int(v) if float(v).is_integer() else v}" for t, v in sorted(axes.items()))
    for rec in font["name"].names:                                   # the names carry the pinned axes
        text = rec.toUnicode()
        if rec.nameID in (1, 4, 16):
            rec.string = f"{text} {suffix}"
        elif rec.nameID == 6:
            rec.string = f"{text}-{suffix}"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    font.save(out)
    return loc


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or len(argv) < 2:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    src = argv[0]
    try:
        if argv[1] == "--axes":
            for tag, lo, d, hi in axes_of(src):
                print(f"{tag}\tmin {lo}\tdefault {d}\tmax {hi}")
            return 0
        out, pins = argv[1], {}
        for a in argv[2:]:
            tag, _, v = a.partition("=")
            if not tag or not v:
                raise ValueError(f"{a!r}: give TAG=value")
            pins[tag] = float(v)
        print(instantiate(src, out, pins))
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
