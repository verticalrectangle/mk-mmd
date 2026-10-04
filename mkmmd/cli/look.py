"""mk look: render quick views to look at (cut, cameras, orbits around a target), with strips, sheets, A/B and guides."""
import argparse
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .. import bridge
from .. import config as CFG
from ..ref.photos import ascii_text
from .common import add_project_arg, emit, get_project, parse_frames, scene_path, UsageError

PRESETS = {"front": (0, 8), "3q": (35, 12), "left": (90, 8), "back": (180, 10), "right": (-90, 8),
           "3q_right": (-35, 12), "3q_back": (145, 15), "top": (0, 80), "low": (0, -20)}

HELP = """Render quick views of a scene to look at. Nothing is saved to the .blend.

Views:
  (default)          the cut: the scene camera with its timeline markers, once per output aspect of mk.toml
  --cam NAME         through a named camera
  --view LIST        preset directions around --target, relative to the cast member's facing:
                     """ + " ".join(PRESETS) + """
                     or yaw:elev pairs such as 20:15 (degrees; yaw 0 = in front, 90 = the model's left)
  Shots of the cut with a render-time look (`style = "silhouette"`, `reflection = {...}` in [[shot]]) are drawn in it,
  as `mk render` draws them; --no-styles draws every shot as it is lit.

Layout: one image per frame, view and aspect; --sheet adds a contact sheet (rows = view x aspect, columns =
frames), --strip one row per view, --ab OTHER.blend renders the same views there and pairs them side by side,
--guides draws thirds and the 5 % safe area. --ref PHOTO[,PHOTO...] writes ref.jpg: each reference photo beside a
rendered view at the render's height, to compare a model with the real thing at the same angle (photo i goes with
view i; one photo is repeated beside every view and one view beside every photo; a view is its first frame and size).

Examples:
  mk look --frames 200,400,600 --sheet                       # the cut, every aspect
  mk look --view front,left,back --target 'bone("head").head' --dist 0.6 --frames 300 --sheet
  mk look --frames t=0:1.3:0.3 --sheet --no-styles           # the lightning shot as it is lit, not as the flat look
  mk look --view 3q --target 'bone("elbow.L").head' --dist 0.35 --lens 60 --frames 189:189
  mk look --cam CloseUp --frames t=2:4:0.5 --strip --engine workbench
  mk look --frames 1 --view 35:12,90:8 --target '(0, 0, 0.7)' --dist 7 --ref refs/car/front_left.jpg,refs/car/side.jpg
"""


def add(sub):
    p = sub.add_parser("look", help="render quick views (cut, cameras, orbits); sheets and A/B", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", nargs="?", metavar="SCENE.blend")
    p.add_argument("--frames", required=True)
    p.add_argument("--cam", action="append", metavar="NAME", help="named camera (repeatable)")
    p.add_argument("--view", help="comma list of presets or yaw:elev pairs (orbit views around --target)")
    p.add_argument("--target", help="expression for the orbit target (default: the cast's head)")
    p.add_argument("--dist", type=float, default=1.0, help="orbit distance (m, default 1)")
    p.add_argument("--lens", type=float, default=50.0, help="orbit lens (mm, default 50)")
    p.add_argument("--cast", help="cast member: whose facing the presets follow and whose head is the default target")
    p.add_argument("--output", action="append", metavar="NAME", help="output aspect(s) from mk.toml (default: all "
                   "for the cut and cameras, square for orbits)")
    p.add_argument("--size", type=int, default=640, help="longest image side in pixels (default 640)")
    p.add_argument("--engine", choices=("eevee", "workbench", "cycles"), default="eevee")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--hide", action="append", default=[], metavar="OBJECT", help="hide an object (repeatable)")
    p.add_argument("--no-styles", action="store_true", help="draw every shot as it is lit, ignoring its silhouette / "
                   "reflection look (the cut shows them by default, as `mk render` draws them)")
    p.add_argument("--sheet", action="store_true", help="also write a contact sheet")
    p.add_argument("--strip", action="store_true", help="also write one strip per view")
    p.add_argument("--ab", metavar="OTHER.blend", help="render the same views on another scene and pair them")
    p.add_argument("--ref", action="append", metavar="PHOTO[,PHOTO...]", help="reference photos to show beside the "
                   "rendered views in ref.jpg (same height; mk ref photos fetches licensed ones)")
    p.add_argument("--guides", action="store_true", help="draw thirds and the safe area")
    p.add_argument("--out", help="folder (default: <project>/.mk/look/<time> or ~/.cache/mk/look/<time>)")
    add_project_arg(p)
    p.set_defaults(func=run)


def _sizes(names, proj, square, longest):
    def fit(w, h):
        s = longest / max(w, h)
        return max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)
    if names:
        if not proj:
            raise UsageError("--output needs a project")
        return [dict(name=o.name, **dict(zip(("w", "h"), fit(*o.size)))) for o in map(proj.output, names)]
    if square or not proj or not proj.outputs:
        return [{"name": "sq", "w": longest, "h": longest}]
    return [dict(name=o.name, **dict(zip(("w", "h"), fit(*o.size)))) for o in proj.outputs]


def _font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def guides(path, safe=0.05):
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    for k in (1, 2):
        d.line([(w * k / 3, 0), (w * k / 3, h)], fill=(156, 207, 216), width=1)
        d.line([(0, h * k / 3), (w, h * k / 3)], fill=(156, 207, 216), width=1)
    d.rectangle([w * safe, h * safe, w * (1 - safe), h * (1 - safe)], outline=(246, 193, 119), width=2)
    im.save(path, quality=90)


def grid(rows, out, label_px=16):
    """rows: [(row label, [(cell label, path), ...])] -> one image."""
    font = _font(label_px)
    cells = [[Image.open(p) for _, p in r] for _, r in rows]
    cw = max(im.width for r in cells for im in r)
    ch = max(im.height for r in cells for im in r)
    pad, top, left = 6, label_px + 8, max(int(font.getlength(lbl)) for lbl, _ in rows) + 16
    ncol = max(len(r) for r in cells)
    sheet = Image.new("RGB", (left + ncol * (cw + pad), len(rows) * (ch + top + pad)), (35, 33, 54))
    d = ImageDraw.Draw(sheet)
    for i, ((lbl, items), ims) in enumerate(zip(rows, cells)):
        y = i * (ch + top + pad)
        d.text((6, y + top + ch // 2 - label_px // 2), lbl, fill=(224, 222, 244), font=font)
        for j, ((clbl, _), im) in enumerate(zip(items, ims)):
            x = left + j * (cw + pad)
            d.text((x, y + 2), clbl, fill=(144, 140, 170), font=font)
            sheet.paste(im, (x, y + top))
    sheet.save(out, quality=90)
    return str(out)


def ref_paths(specs, proj):
    """The reference images named by --ref values (comma lists, repeatable): relative paths are tried from the working
    directory, then from the project folder."""
    out = []
    for spec in specs or []:
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            p = Path(item).expanduser()
            if not p.is_absolute() and not p.exists() and proj:
                p = proj.root / p
            if not p.is_file():
                raise UsageError(f"--ref: no such image {item!r}")
            out.append(p)
    if not out:
        raise UsageError("--ref needs at least one image")
    return out


def pair_refs(refs, shots):
    """[(photo, shot)] rows for the side-by-side sheet: photo i beside view i, where a view is its first frame and size;
    the shorter list repeats (one photo beside every view, one view beside every photo)."""
    views = {}
    for s in shots:
        views.setdefault(s["view"], s)
    views = list(views.values())
    return [(refs[i % len(refs)], views[i % len(views)]) for i in range(max(len(refs), len(views)))]


def side_by_side(pairs, out, label_px=14, pad=8):
    """One image, a row per (photo, shot): the photo scaled to the render's height beside it, labelled above."""
    font, top = _font(label_px), label_px + 8
    rows = []
    for ref, shot in pairs:
        render = Image.open(shot["path"]).convert("RGB")
        photo = Image.open(ref).convert("RGB")
        photo = photo.resize((max(1, round(photo.width * render.height / photo.height)), render.height), Image.LANCZOS)
        rows.append((ref, shot, photo, render))
    sheet = Image.new("RGB", (max(p.width + r.width + 3 * pad for _, _, p, r in rows),
                              sum(r.height + top + pad for _, _, _, r in rows) + pad), (35, 33, 54))
    d, y = ImageDraw.Draw(sheet), pad
    for ref, shot, photo, render in rows:
        d.text((pad, y), "reference: " + ascii_text(ref.name)[:60], fill=(246, 193, 119), font=font)
        d.text((2 * pad + photo.width, y), ascii_text(f"{shot['view']} {shot['size']} frame {shot['frame']}"),
               fill=(224, 222, 244), font=font)
        sheet.paste(photo, (pad, y + top))
        sheet.paste(render, (2 * pad + photo.width, y + top))
        y += render.height + top + pad
    sheet.save(out, quality=90)
    return str(out)


def run(args):
    proj = get_project(args)
    blend = scene_path(args.scene, proj)
    frames = parse_frames(args.frames, proj)
    refs = ref_paths(args.ref, proj) if args.ref else []
    cast_arm = None
    if args.cast or (proj and len(proj.cast) == 1):
        cast_arm = proj.cast_member(args.cast).get("armature") if proj else None
    views = []
    for c in args.cam or []:
        views.append({"name": c, "kind": "camera", "camera": c})
    if args.view:
        target = args.target or (f'bone("head", {cast_arm!r}).head' if cast_arm else 'bone("head").head')
        for spec in args.view.split(","):
            spec = spec.strip()
            if spec in PRESETS:
                yaw, elev = PRESETS[spec]
            elif ":" in spec:
                yaw, elev = (float(x) for x in spec.split(":"))
            else:
                raise UsageError(f"unknown view {spec!r} (presets: {', '.join(PRESETS)}, or yaw:elev)")
            views.append({"name": spec.replace(":", "_"), "kind": "orbit", "target": target, "facing": cast_arm,
                          "yaw": yaw, "elev": elev, "dist": args.dist, "lens": args.lens})
    if not views:
        views.append({"name": "cut", "kind": "shot"})
    sizes = _sizes(args.output, proj, square=bool(args.view) and not args.cam, longest=args.size)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out).expanduser() if args.out else (
        (proj.mk_dir / "look" / stamp) if proj else CFG.cache_dir() / "look" / stamp)
    job = {"frames": frames, "views": views, "sizes": sizes, "engine": args.engine, "samples": args.samples,
           "hide": args.hide, "armature": cast_arm, "styles": not args.no_styles}
    t0 = time.time()
    scenes = [("a", blend)] + ([("b", Path(args.ab).expanduser().resolve())] if args.ab else [])
    shots = {}
    for tag, sc in scenes:
        shots[tag] = bridge.run("look", {**job, "out": str(out / tag if args.ab else out)}, blend=sc, project=proj,
                                timeout=3600)
    if args.guides:
        for lst in shots.values():
            for s in lst:
                guides(s["path"])
    res = {"scene": str(blend), "out": str(out), "seconds": round(time.time() - t0, 1),
           "images": [s["path"] for s in shots["a"]]}
    if args.ab:
        res["images_b"] = [s["path"] for s in shots["b"]]
        pairs = []
        for sa, sb in zip(shots["a"], shots["b"]):
            pairs.append((f"{sa['view']} {sa['size']} {sa['frame']}", [("A", sa["path"]), ("B", sb["path"])]))
        res["ab"] = grid(pairs, out / "ab.jpg")
    if refs:
        pairs = pair_refs(refs, shots["a"])
        res["ref"] = side_by_side(pairs, out / "ref.jpg")
        res["ref_pairs"] = [{"photo": str(r), "view": s["view"], "frame": s["frame"]} for r, s in pairs]
    if args.sheet or args.strip:
        rows = {}
        for s in shots["a"]:
            rows.setdefault(f"{s['view']} {s['size']}", []).append((str(s["frame"]), s["path"]))
        if args.sheet:
            res["sheet"] = grid(list(rows.items()), out / "sheet.jpg")
        if args.strip:
            res["strips"] = [grid([(k, v)], out / f"strip_{k.replace(' ', '_')}.jpg") for k, v in rows.items()]
    emit(res)
    return 0
