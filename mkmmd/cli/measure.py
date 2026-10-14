"""mk measure: read measurements off a pixel map (mk look --map): edges, extents, gaps, points and distances in pixels and
centimetres."""
import argparse

from .. import pixmap as PX
from .common import UsageError, emit

HELP = """Read measurements off a pixel map, the <image>.map.npz `mk look --map` writes beside an image: what is at each
of its pixels (material, object, the point in the world and on the model at rest, its depth), from the scene's own
geometry seen through the image's camera. Pixels are (x, y) from the image's top left; ranges are inclusive, A:B or
A:B:STEP. Materials are named as in the scene (a comma list; `*` is any surface); a name the map lacks is an error that
lists the ones it has.

Without a question: the image size, the camera and every material seen (pixels, box).
  --at X,Y               what a pixel shows: material, object, polygon, world and rest point, depth, cm per pixel
  --edges MATS --x A:B   per column, the first and last row showing MATS (--y keeps to some rows): pixels, the size as
                         seen (seen_cm) and between the two surface points (cm in the world, cm_rest at rest), and the
                         runs (2 or more: something crosses the material)
  --extent MATS --y A:B  per row, the first and last column showing MATS (--x keeps to some columns)
  --gaps [MATS] --y A:B  per row, the runs that show no surface (or none of MATS) with one on both sides: the space
                         between an arm and the body, an opening (--x keeps to some columns)
  --dist X,Y:X,Y         centimetres between the surface points at two pixels, in the world and at rest
  --where X,Y,Z          where a world point (metres) is in the image, and whether a surface hides it

Examples:
  mk measure front_sq_00061.map.npz
  mk measure front_sq_00061.map.npz --edges セーターリブ --x 300:900:20 --y 150:500
  mk measure front_sq_00061.map.npz --gaps --y 430:600:10 --x 150:480
  mk measure front_sq_00061.map.npz --at 600,300 --dist 600,254:600,368 --where 0.098,0.006,1.286
"""


def add(sub):
    p = sub.add_parser("measure", help="measurements off a pixel map (mk look --map): edges, gaps, points, cm",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map", metavar="MAP.npz", help="a pixel map (or its image: the map beside it is read)")
    p.add_argument("--at", action="append", default=[], metavar="X,Y", help="what a pixel shows (repeatable)")
    p.add_argument("--edges", metavar="MATS", help="first and last row of the materials per column (needs --x)")
    p.add_argument("--extent", metavar="MATS", help="first and last column of the materials per row (needs --y)")
    p.add_argument("--gaps", nargs="?", const="*", metavar="MATS", help="runs between covered pixels per row (needs "
                   "--y; MATS: what counts as covered, default any surface)")
    p.add_argument("--x", metavar="A:B[:STEP]", help="columns (inclusive)")
    p.add_argument("--y", metavar="A:B[:STEP]", help="rows (inclusive)")
    p.add_argument("--dist", action="append", default=[], metavar="X,Y:X,Y", help="cm between two pixels' surface "
                   "points (repeatable)")
    p.add_argument("--where", action="append", default=[], metavar="X,Y,Z", help="a world point in the image "
                   "(repeatable)")
    p.set_defaults(func=run)


def _pixel(text, what):
    try:
        x, y = (round(float(v)) for v in text.split(","))
    except ValueError:
        raise UsageError(f"{what} {text!r}: want X,Y (pixels)") from None
    return x, y


def run(args):
    path = PX.map_path(args.map) if not str(args.map).endswith(".npz") else args.map
    try:
        m = PX.Map.load(path)
        xs = PX.parse_span(args.x, m.w, "--x") if args.x else None
        ys = PX.parse_span(args.y, m.h, "--y") if args.y else None
        res = {"map": str(path), "size": [m.w, m.h]}
        asked = False
        if args.at:
            res["at"] = [m.at(*_pixel(a, "--at")) for a in args.at]
            asked = True
        if args.edges:
            if xs is None:
                raise UsageError("--edges needs --x (the columns)")
            res["edges"] = {"materials": args.edges, "columns": m.edges(args.edges, xs, ys)}
            asked = True
        if args.extent:
            if ys is None:
                raise UsageError("--extent needs --y (the rows)")
            res["extent"] = {"materials": args.extent, "rows": m.extent(args.extent, ys, xs)}
            asked = True
        if args.gaps:
            if ys is None:
                raise UsageError("--gaps needs --y (the rows)")
            res["gaps"] = {"materials": args.gaps, "rows": m.gaps(ys, None if args.gaps == "*" else args.gaps, xs)}
            asked = True
        for d in args.dist:
            a, _, b = d.partition(":")
            res.setdefault("dist", []).append(m.distance(_pixel(a, "--dist"), _pixel(b, "--dist")))
            asked = True
        for w in args.where:
            try:
                pt = [float(v) for v in w.split(",")]
            except ValueError:
                pt = []
            if len(pt) != 3:
                raise UsageError(f"--where {w!r}: want X,Y,Z (metres)")
            res.setdefault("where", []).append(m.where(pt))
            asked = True
        if not asked:
            res.update(m.summary())
    except PX.MapError as e:
        raise UsageError(str(e)) from None
    emit(res)
    return 0
