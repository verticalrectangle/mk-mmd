"""mk mark: draw on an image: a pixel grid to read positions, a crop and a zoom in the image's own pixels, a plan from a
marks file, a photo beside it."""
import argparse
from pathlib import Path

from PIL import Image

from .. import marks as MK
from .. import pixmap as PX
from .common import UsageError, emit

HELP = """Draw on an image, to read positions off it or to show a plan on it.

  --grid [STEP]          a grid of the image's own pixels, a line every STEP (default 50), labelled every other line
  --crop X0,Y0,X1,Y1     keep that part (pixels; X1, Y1 excluded), --zoom K scales it: the grid and the marks keep the
                         image's own coordinates
  --marks FILE.toml      a plan: lines and curves (dashed, smooth), fills (between two lines), arrows, circles, boxes and
                         text; a point is [x, y] in the image's pixels or [x, y, z] in the world's metres (projected
                         through the image's pixel map, mk look --map: --map, default the one beside the image)
  --ref PHOTO            the photo on the left at the same height, to compare with

The marks file (TOML, one array of tables per kind):
  [[line]]    points, color, width (3), dashed, smooth, closed, name (for fills)
  [[fill]]    points or between = ["line a", "line b"], color, opacity (0.35), smooth
  [[arrow]]   from, to, color, width
  [[circle]]  at, radius, color, width
  [[box]]     box = [x, y, w, h], color, width
  [[text]]    at (top left), text, color (white), size (20), back ("#28203c", "none"), opacity (0.88)
Fills are drawn first, text last. Widths and sizes are pixels of the finished picture.

Examples:
  mk mark front_sq_00061.jpg --grid                                   # read positions
  mk mark front_sq_00061.jpg --grid 25 --crop 250,175,600,675 --zoom 2 # a close look, same coordinates
  mk mark front_sq_00061.jpg --marks plan.toml --ref photo.png --out plan.png
"""


def add(sub):
    p = sub.add_parser("mark", help="draw on an image: a pixel grid, crops, a plan from a marks file, a photo beside",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", metavar="IMAGE")
    p.add_argument("--marks", metavar="FILE.toml", help="the marks to draw")
    p.add_argument("--grid", nargs="?", const=50, type=int, metavar="STEP", help="a pixel grid (default every 50 px)")
    p.add_argument("--crop", metavar="X0,Y0,X1,Y1", help="the part of the image to keep (pixels)")
    p.add_argument("--zoom", type=float, default=1.0, help="scale of the kept part (default 1)")
    p.add_argument("--map", metavar="MAP.npz", help="the image's pixel map, for [x, y, z] points (default: beside it)")
    p.add_argument("--ref", metavar="PHOTO", help="a photo to put on the left, at the same height")
    p.add_argument("--out", help="the picture (default: <image>.marked.png beside the image)")
    p.set_defaults(func=run)


def run(args):
    src = Path(args.image).expanduser()
    if not src.is_file():
        raise UsageError(f"no such image {str(src)!r}")
    if not (args.marks or args.grid or args.crop or args.ref):
        raise UsageError("nothing to draw: give --marks, --grid, --crop or --ref")
    if args.zoom <= 0:
        raise UsageError("--zoom must be positive")
    if args.grid is not None and args.grid <= 0:
        raise UsageError("--grid STEP must be positive")
    try:
        box = None
        if args.crop:
            try:
                box = [round(float(v)) for v in args.crop.split(",")]
            except ValueError:
                box = []
            if len(box) != 4:
                raise UsageError(f"--crop {args.crop!r}: want X0,Y0,X1,Y1 (pixels)")
        with Image.open(src) as im:
            im = im.convert("RGB")
        pic, view = MK.crop(im, box, args.zoom)
        marks = MK.load(args.marks) if args.marks else {}
        mp = Path(args.map).expanduser() if args.map else PX.map_path(src)
        if args.map or mp.is_file():
            try:
                view.project = PX.Map.load(mp).cam.project
            except PX.MapError:
                if args.map:
                    raise
        if args.grid:
            pic = MK.grid(pic, view, args.grid)
        if marks:
            pic = MK.draw(pic, marks, view)
        if args.ref:
            ref = Path(args.ref).expanduser()
            if not ref.is_file():
                raise UsageError(f"--ref: no such image {str(ref)!r}")
            with Image.open(ref) as ph:
                pic = MK.beside(ph, pic)
    except (MK.MarkError, PX.MapError) as e:
        raise UsageError(str(e)) from None
    out = Path(args.out).expanduser() if args.out else src.with_name(src.stem + ".marked.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    pic.save(out)
    emit({"image": str(src), "out": str(out), "size": list(pic.size),
          "marks": {k: len(v) for k, v in marks.items()}, "crop": box, "zoom": args.zoom})
    return 0
