"""mk model: build an original character model from a spec (part builders in code -> PMX -> rig.json -> review .blend)."""
import argparse
import json
import os
import random
import re
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from .. import bridge
from .. import config as CFG
from ..model import assemble as AS
from ..model import build as BD
from ..model import pmx_io
from ..model import spec as SP
from . import doctor
from .common import CHECK_FAILED, UsageError, emit

HELP = """Build a character from a spec: part builders (body, head, hair, outfit, ...) are plain Python in
mkmmd/model/parts/, driven by a TOML spec; the parts are merged, exported as a PMX (textures next to it), imported
back with mmd_tools, verified against what was assembled (positions, weights, morphs, UVs, rigid bodies; 0.1 mm), and
described like `mk inspect` does (rig.json). A review .blend with a neutral studio works with `mk look`. The spec's
[model] table, the part builders and the conventions are in docs/design.md: Characters (mk model). Exit 1 when a
builder, the assembly or the verification fails.

Examples:
  mk model new mika                                      # ./mika/model.toml: the girl base, ready to change
  mk model build ~/projects/rin/model.toml
  mk model build model.toml --only head                  # head + the parts it needs, into <out>/only_head/
  mk model build model.toml --no-export                  # run and check the builders only (no Blender)
  mk model build model.toml --set hair.length=0.3 --out /tmp/try
  mk look <out>/<name>.blend --view front,3q --target "bone('head').head" --dist 1.2 --frames 1
  mk model info model.toml                               # what would be built, in which order

Output files in <out>: <name>.pmx, tex/*.png, <name>.blend, <name>.rig.json, build.json (the printed report).
"""

NEW_HELP = """Start a character from a model base shipped with mk (mkmmd/model/bases/): writes DIR/model.toml, whose
[model] includes the base, so the character is the base plus the tables written under it. Change it one table at a time
and build; docs/model_base.md lists the tables, what they change, and how to check the result. Bases: {bases}.

Examples:
  mk model new mika                                      # ./mika/model.toml on the girl base
  mk model new mika --from rin --dir ~/chars/mika        # start from the worked example instead
  mk model build mika/model.toml
"""

NEW_SPEC = """# {name}: a character on the "{base}" model base (mkmmd/model/bases/{base}/). The base comes first; every table
# written below goes over it key by key, so write only what changes. docs/model_base.md lists the tables and what they
# change; mkmmd/model/bases/rin/ is a worked example (the girl base plus Rin's hair, ears, tails, dress and colours).
#   mk model build model.toml            # PMX, textures, rig.json and a review .blend in [model] out
[model]
name = "{name}"
include = ["base:{base}"]
out = "{out}"

# For example (uncomment and change):
# [colors.hair]
# base = "#3b2a2f"
# [hair.braids]
# enabled = true
"""

LAB_HELP = """Look at models without Blender: views and pose sheets posed with each model's own weights, several models
side by side at one scale, and their numbers (lengths, widths and girths in mm, cut across the skin under any clothes).
A MODEL is a spec (model.toml or base:NAME: its parts are built and assembled in-process, as mk model build does) or a
.pmx. Writes SHEET.png, SHEET.json (each cell's camera, the numbers) and SHEET.mask.png (what mk model trace measures
against). Regions: {regions}. Poses: {poses}, or any of the model's morph names (one row each).

Examples:
  mk model lab base:girl --region hand --poses rest,relaxed,fist,spread
  mk model lab model.toml ~/mk-assets/models/rin_mk/rin.pmx --parts all --views front,outer,3q
  mk model lab model.toml --region head --poses rest,ω,口角上げ --out /tmp/mouth.png
"""

TRACE_HELP = """Read a red line drawn on a lab sheet back in millimetres. Open the sheet, take a screenshot of the part you
mean (zoomed or not; window borders are fine), draw on it in pure red, save it. Prints the cell, the line's points in
model space, and how far inside (+: trim this much) or outside (-: add) the outline it runs, every 2 mm along it, with
its place along the region (from the wrist for a hand, from the floor for a body).

Example:
  mk model trace ~/Pictures/marked.png --sheet ~/mk-assets/models/girl/lab/hand_L.png
"""

GLB_HELP = """Write a model, posed, as one .glb with its textures inside: what Tern's 3D block (and any glTF viewer) turns,
pans and zooms. No Blender: a .pmx is read and posed with its own weights; a spec is built in-process, as mk model build
does. Poses: {poses}; --morph puts morphs on top (NAME or NAME=WEIGHT, repeatable). Prints the model's numbers as JSON
(names, height, counts, its morphs by panel, the poses its bones allow); --info prints them and writes nothing.

Examples:
  mk model glb ~/mk-assets/models/rin_mk/rin.pmx --out /tmp/rin.glb          # T-pose
  mk model glb base:girl --parts body --pose arms_down --out /tmp/girl.glb
  mk model glb rin.pmx --pose rest --morph まばたき --morph 笑い=0.5 --out /tmp/smile.glb
  mk model glb base:girl --parts head --pose rest --region head --out /tmp/face.glb   # the face alone, up close
  mk model glb rin.pmx --info
"""


def add(sub):
    p = sub.add_parser("model", help="build an original character (parts -> PMX, rig.json, review .blend)",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ss = p.add_subparsers(dest="model_command", metavar="ACTION", required=True)
    b = ss.add_parser("build", help="build the parts, export the PMX, import it back and verify it",
                      description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    b.add_argument("spec", metavar="SPEC.toml", help="the model spec (model.toml): [model] plus the part tables")
    b.add_argument("--only", metavar="PARTS", help="comma list of parts to build, plus the parts they need")
    b.add_argument("--out", metavar="DIR", help="output folder (default: [model] out; with --only: <out>/only_<parts>)")
    b.add_argument("--no-export", action="store_true", help="only run and check the builders (no PMX, no Blender)")
    b.add_argument("--blend", action="store_true", help="write the review .blend (default when exporting)")
    b.add_argument("--no-blend", action="store_true", help="skip the review .blend")
    b.add_argument("--no-verify", action="store_true", help="skip the import-back verification")
    b.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override a spec value (TOML syntax, repeatable): hair.length=0.3")
    b.add_argument("--full", action="store_true", help="also print the part-by-part warnings and the whole morph list")
    b.set_defaults(func=run_build)
    i = ss.add_parser("info", help="print what a build would do (parts in order, output folder, builders)")
    i.add_argument("spec", metavar="SPEC.toml", help="the model spec (model.toml)")
    i.add_argument("--only", metavar="PARTS", help="comma list of parts: show the plan for these and what they need")
    i.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override a spec value (TOML syntax, repeatable): hair.length=0.3")
    i.set_defaults(func=run_info)
    s = ss.add_parser("studio", help="copy a built scene with the neutral review studio (grey world, floor, lights)")
    s.add_argument("scene", metavar="SCENE.blend", help="a scene with models in it (a built project, several imported "
                                                        "PMX); it is left alone")
    s.add_argument("--out", required=True, metavar="OUT.blend", help="where to save the copy that has the studio")
    s.add_argument("--floor", action="append", default=[], metavar="X,Y", help="a floor disc at x,y (repeatable)")
    s.add_argument("--lights", action="store_true", help="add key/fill/rim suns (default: the scene has its own)")
    s.add_argument("--height", type=float, default=1.6, help="model height for aiming the lights")
    s.set_defaults(func=run_studio)

    n = ss.add_parser("new", help="start a character on a model base (a folder with a model.toml that includes it)",
                      description=NEW_HELP.format(bases=", ".join(SP.bases()) or "none"),
                      formatter_class=argparse.RawDescriptionHelpFormatter)
    n.add_argument("name", help="the character's name: the PMX model name and the output file stem")
    n.add_argument("--from", dest="base", default="girl", metavar="BASE", help="the base to start from (default: girl)")
    n.add_argument("--dir", metavar="DIR", help="the folder to create (default: ./NAME)")
    n.add_argument("--out", metavar="DIR", help="the character's [model] out (default: ~/mk-assets/models/NAME)")
    n.set_defaults(func=run_new)

    from ..model import lab as LAB
    lb = ss.add_parser("lab", help="views, pose sheets and side-by-side numbers of models (no Blender)",
                       description=LAB_HELP.format(regions=", ".join(LAB.REGIONS), poses=", ".join(LAB.POSES)),
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    lb.add_argument("models", nargs="+", metavar="MODEL", help="a spec (model.toml, base:NAME) or a .pmx; several are "
                                                               "shown side by side at one scale")
    lb.add_argument("--region", default="body", choices=list(LAB.REGIONS), help="what to frame (default: body)")
    lb.add_argument("--side", default="L", choices=["L", "R"], help="her side for hand, foot, arm, leg (default: L)")
    lb.add_argument("--views", metavar="V,..", help="comma list (default per region; body: front,outer,back,3q; "
                                                    "hand: back,palm,thumb,3q; also inner, top, sole, little, tip)")
    lb.add_argument("--poses", default="rest", metavar="P,..", help="comma list of poses or morph names, one row each")
    lb.add_argument("--parts", metavar="PARTS", help="parts to build for a spec: a comma list, or all (default: body; "
                                                     "head for the head region)")
    lb.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                    help="override a value of every spec (TOML syntax, repeatable): body.hand.length=0.15")
    lb.add_argument("--label", action="append", default=[], help="a model's label on the sheet (repeatable, in order)")
    lb.add_argument("--size", type=int, default=360, help="cell size in pixels (default: 360)")
    lb.add_argument("--unit", type=float, default=0.08, help="metres per PMX unit of .pmx models (default: 0.08)")
    lb.add_argument("--out", metavar="SHEET.png", help="the sheet (default: <first model's folder or [model] out>/lab/"
                                                       "<region>[_<side>].png)")
    lb.set_defaults(func=run_lab)
    tr = ss.add_parser("trace", help="a red line drawn on a lab sheet (or a screenshot of part of one) in millimetres",
                       description=TRACE_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    tr.add_argument("marked", metavar="MARKED.png", help="the sheet, or a screenshot of part of it, with a pure red line")
    tr.add_argument("--sheet", required=True, metavar="SHEET.png", help="the lab sheet (its .json and .mask.png beside it)")
    tr.set_defaults(func=run_trace)
    gl = ss.add_parser("glb", help="a model posed (T-pose by default) as one .glb with its textures (no Blender)",
                       description=GLB_HELP.format(poses=", ".join(LAB.POSES)),
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    gl.add_argument("model", metavar="MODEL", help="a spec (model.toml, base:NAME) or a .pmx")
    gl.add_argument("--pose", default="tpose", choices=list(LAB.POSES), help="the pose (default: tpose)")
    gl.add_argument("--morph", action="append", default=[], metavar="NAME[=WEIGHT]", help="a morph on top (repeatable)")
    gl.add_argument("--parts", metavar="PARTS", help="parts to build for a spec: a comma list, or all (default: all)")
    gl.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                    help="override a spec value (TOML syntax, repeatable)")
    gl.add_argument("--region", default="body", choices=list(LAB.REGIONS), help="what to write: body (all, the "
                    "default), or a semantic bone's subtree: head (the face up close), hand, foot, arm, leg")
    gl.add_argument("--side", default="L", choices=["L", "R"], help="her side for hand, foot, arm, leg (default: L)")
    gl.add_argument("--unit", type=float, default=0.08, help="metres per PMX unit of a .pmx (default: 0.08)")
    gl.add_argument("--out", metavar="FILE.glb", help="where to write it (default: ./<model>-<pose>.glb)")
    gl.add_argument("--info", action="store_true", help="print the model's numbers only")
    gl.set_defaults(func=run_glb)


def _lab_parts(args):
    if args.parts and args.parts.strip() == "all":
        return None
    if args.parts:
        return [s.strip() for s in args.parts.split(",") if s.strip()]
    return ["head"] if args.region == "head" else ["body"]


def run_lab(args):
    from ..model import lab as LAB
    t0 = time.time()
    parts = _lab_parts(args)
    models, loaded = [], []
    for i, src in enumerate(args.models):
        t = time.time()
        try:
            m = LAB.load(src, label=args.label[i] if i < len(args.label) else None, overrides=args.overrides,
                         parts=None if src.lower().endswith(".pmx") else parts, unit=args.unit)
        except (SP.SpecError, BD.BuildError, ValueError) as e:
            raise UsageError(f"{src}: {e}")
        while m.label in [x.label for x in models]:          # two models of one name: number the later ones
            m.label = f"{m.label} {len(models) + 1}"
        models.append(m)
        loaded.append({"label": m.label, "source": src, "vertices": len(m.V), "triangles": len(m.T),
                       "seconds": round(time.time() - t, 1)})
    views = [v.strip() for v in args.views.split(",") if v.strip()] if args.views else None
    poses = [p.strip() for p in args.poses.split(",") if p.strip()]
    try:
        img, layout, cover = LAB.sheet(models, region=args.region, side=args.side, views=views, poses=poses,
                                       size=args.size)
    except ValueError as e:
        raise UsageError(str(e))
    if args.out:
        out = Path(args.out).expanduser()
    else:
        first = args.models[0]
        folder = (Path(first).expanduser().parent if first.lower().endswith(".pmx")
                  else SP.model_cfg(SP.load(first, args.overrides))["out"])
        out = folder / "lab" / (f"{args.region}_{args.side}.png" if args.region in LAB.SIDED else f"{args.region}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    cover.save(out.with_name(out.stem + ".mask.png"))
    out.with_suffix(".json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"sheet": str(out), "layout": str(out.with_suffix(".json")), "cells": len(layout["cells"]), "models": loaded,
          "numbers": layout["numbers"], "seconds": round(time.time() - t0, 1)})
    return 0


def run_trace(args):
    from ..model import lab as LAB
    sheet, marked = Path(args.sheet).expanduser(), Path(args.marked).expanduser()
    layout_path = sheet.with_suffix(".json")
    for p in (sheet, marked):
        if not p.is_file():
            raise UsageError(f"{p} does not exist")
    if not layout_path.is_file():
        raise UsageError(f"{layout_path} does not exist: trace reads sheets made by mk model lab")
    try:
        emit(LAB.trace(marked, sheet, json.loads(layout_path.read_text(encoding="utf-8"))))
    except ValueError as e:
        raise UsageError(str(e))
    return 0


PANELS = {0: "system", 1: "eyebrow", 2: "eye", 3: "mouth", 4: "other"}


def model_numbers(m):
    """What a viewer shows of a lab Model: its names, height, counts, morphs by panel and the poses its bones allow."""
    from ..model import lab as LAB
    z = m.V[m.visible, 2] if m.visible.any() else m.V[:, 2]
    return {"name": m.names[0] or m.label, "name_en": m.names[1], "label": m.label, "source": m.source,
            "height_m": round(float(z.max() - z.min()), 3) if len(z) else 0.0, "vertices": int(len(m.V)),
            "triangles": int(len(m.T)), "materials": len(m.mat_names), "drawn_materials": int((~m.hidden).sum()),
            "bones": len(m.bones), "morphs": [{"name": k, "panel": PANELS.get(p, "other")} for k, p in m.morph_panels.items()],
            "poses": ["rest"] + [p for p in LAB.POSES if p != "rest" and LAB.pose_rotations(m, p)]}


def run_glb(args):
    from ..model import glb as GLB
    from ..model import lab as LAB
    t0 = time.time()
    pmx = args.model.lower().endswith(".pmx")
    parts = (None if pmx or not args.parts or args.parts.strip() == "all"
             else [s.strip() for s in args.parts.split(",") if s.strip()])
    morphs = {}
    for item in args.morph:
        name, eq, w = item.partition("=")
        try:
            morphs[name.strip()] = float(w) if eq else 1.0
        except ValueError:
            raise UsageError(f"--morph {item!r}: NAME or NAME=WEIGHT")
    with tempfile.TemporaryDirectory(prefix="mk_glb_") as tmp:
        try:
            m = LAB.load(args.model, overrides=args.overrides, parts=parts, unit=args.unit, workdir=None if pmx else tmp)
            rep = model_numbers(m)
            if args.info:
                emit({**rep, "seconds": round(time.time() - t0, 1)})
                return 0
            V, N = m.deform(args.pose, morphs)
            keep = None
            if args.region != "body":
                inside = LAB.region_mask(m, args.region, args.side)          # vertices of the bone's subtree
                keep = inside[m.T].all(axis=1)
                if not keep.any():
                    raise ValueError(f"nothing of the {args.region} region is drawn")
        except (SP.SpecError, BD.BuildError, ValueError) as e:
            raise UsageError(f"{args.model}: {e}")
        out = Path(args.out).expanduser() if args.out else Path(f"{m.label}-{args.pose}.glb")
        glb = GLB.write(out, m, V, N, keep=keep, extras={"source": m.source, "pose": args.pose, "morphs": morphs,
                                                         "region": args.region})
    emit({**rep, "out": str(out.resolve()), "pose": args.pose, "pose_applied": args.pose in rep["poses"],
          "morphs_applied": morphs, "glb": glb, "seconds": round(time.time() - t0, 1)})
    return 0


def run_new(args):
    if args.base not in SP.bases():
        raise UsageError(f"no model base {args.base!r}; bases: {', '.join(SP.bases()) or 'none'}")
    if not re.fullmatch(r"[\w-]+", args.name):
        raise UsageError(f"name {args.name!r}: letters, digits, '_' and '-' only (it is the output file stem)")
    folder = Path(args.dir or args.name).expanduser()
    spec_file = folder / "model.toml"
    if spec_file.exists():
        raise UsageError(f"{spec_file} exists: pick another name or --dir")
    folder.mkdir(parents=True, exist_ok=True)
    spec_file.write_text(NEW_SPEC.format(name=args.name, base=args.base, out=args.out or f"~/mk-assets/models/{args.name}"),
                         encoding="utf-8")
    spec = SP.load(spec_file)
    emit({"created": str(spec_file.resolve()), "base": args.base, "files": [str(f) for f in spec.files],
          "build": f"mk model build {spec_file}"})
    return 0


def run_studio(args):
    scene = Path(args.scene).expanduser()
    if not scene.exists():
        raise UsageError(f"{scene} does not exist")
    floors = [[float(v) for v in f.split(",")] for f in args.floor] or [[0.0, 0.0]]
    res = _studio(scene, args, floors)
    emit(res)
    return 0


def _studio(scene, args, floors):
    for attempt in range(4):
        try:
            return bridge.run("model_studio", {"out": str(Path(args.out).expanduser().resolve()), "lights": args.lights,
                                               "floors": floors, "height": args.height}, blend=scene, timeout=600)
        except bridge.BlenderError as e:
            if attempt == 3 or not any(s in str(e) for s in ("could not be found", "opencc", "No module named")):
                raise
            repair_wheels()


def repair_wheels():
    """Reinstall mmd_tools' bundled wheels when a Blender started with --factory-startup deleted them (what
    `mk doctor --fix` does). Returns the names of the wheels restored."""
    m = re.search(r"blender-(\d+\.\d+)", str(CFG.load()["blender"]))
    return doctor._fix_wheels(m.group(1) if m else "4.2")


def run_blender(op, args, timeout=1800, tries=6):
    """`bridge.run` that repairs and retries when mmd_tools failed to load (missing opencc wheel: another process
    started with --factory-startup, or several Blenders starting at once raced on the wheel folder)."""
    for attempt in range(tries):
        try:
            return bridge.run(op, args, timeout=timeout)
        except bridge.BlenderError as e:
            msg = str(e)
            transient = "could not be found" in msg or "opencc" in msg or "No module named" in msg
            if not transient or attempt == tries - 1:
                raise
            repair_wheels()
            time.sleep(1.0 + 2.0 * random.random())


def _load(args):
    try:
        spec = SP.load(args.spec, args.overrides)
        cfg = SP.model_cfg(spec)
    except SP.SpecError as e:
        raise UsageError(str(e))
    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    return spec, cfg, only


def run_info(args):
    spec, cfg, only = _load(args)
    try:
        plan = BD.plan(spec, only)
    except BD.BuildError as e:
        raise UsageError(str(e))
    out = {"spec": str(spec.path), "files": [str(f) for f in spec.files], "name": cfg["name"], "parts": plan,
           "out": str(cfg["out"]), "seed": cfg["seed"], "needs": {}}
    for n in plan:
        try:
            out["needs"][n] = list(cfg["needs"].get(n) or BD.get_builder(n, spec).needs)
        except BD.BuildError as e:
            out["needs"][n] = f"ERROR: {e}"
    emit(out)
    return 0


def part_stats(part, seconds=None):
    faces = sum(len(m.faces) for m in part.meshes)
    return {"name": part.name, "seconds": seconds, "meshes": len(part.meshes),
            "vertices": sum(len(m.verts) for m in part.meshes), "faces": faces, "bones": len(part.bones),
            "materials": len(part.materials), "morphs": len({k for m in part.meshes for k in m.morphs}),
            "bodies": len(part.bodies), "joints": len(part.joints), "subsurf": sorted({m.subsurf for m in part.meshes
                                                                                         if m.subsurf})}


def rig_report(rig):
    """The parts of rig.json a modeller wants after a build: required bones, morph map, chain families, bodies."""
    fams = {}
    for c in rig["chains"]:
        f = fams.setdefault(c["family"], {"chains": 0, "bones": 0})
        f["chains"] += 1
        f["bones"] += len(c["bones"])
    mapped = set(rig["morphs"].values())
    return {
        "semantic_bones": len(rig["map"]), "missing_required": rig["missing_required"],
        "missing_optional": [s for s in rig["missing"] if s not in rig["missing_required"]],
        "morph_map": rig["morphs"], "morphs_unmapped": [m["name"] for m in rig["morph_list"] if m["name"] not in mapped],
        "chain_families": fams, "bodies": rig["stats"]["bodies"], "dynamic_bodies": rig["stats"]["dynamic_bodies"],
        "joints": rig["stats"]["joints"], "measure": rig["measure"], "quirks": rig["quirks"], "kind": rig.get("kind"),
    }


def _lock(path):
    """Exclusive advisory lock on `path` for the life of the returned handle: two `mk model build` runs into the same
    folder (a modeller and a QA pass) take turns instead of deleting each other's PMX and textures."""
    import fcntl
    fh = open(path, "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def run_build(args):
    t_all = time.time()
    spec, cfg, only = _load(args)
    name = cfg["name"]
    out = Path(args.out).expanduser() if args.out else cfg["out"]
    if only and not args.out:
        out = out / ("only_" + "_".join(only))
    out = out.resolve()
    tex_dir = out / "tex"
    report = {"spec": str(spec.path), "files": [str(f) for f in spec.files], "name": name, "out": str(out),
              "only": only, "seconds": {}}
    out.mkdir(parents=True, exist_ok=True)
    (out / ".mk").mkdir(exist_ok=True)
    lock = _lock(out / ".mk" / "build.lock")           # a second build into this folder waits for the first
    # nothing is deleted up front: other projects may be casting the last good PMX right now; every file is replaced
    # atomically when its new version is ready, and build.json says "running" (ok false) until this build is done
    (out / "build.json").write_text(json.dumps({"ok": False, "stage": "running", "pid": os.getpid()}), encoding="utf-8")

    def finish(rep):
        """Print the report and keep it as build.json (failures too: it tells whether the files here are current)."""
        (out / "build.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        emit(rep)

    # ---- 1. the parts (numpy + PIL, no Blender)
    t0 = time.time()
    ctx = BD.BuildCtx(spec, tex_dir, seed=cfg["seed"])
    try:
        parts = BD.run(spec, only=only, ctx=ctx)
    except BD.BuildError as e:
        report.update(ok=False, stage="parts", error=str(e), trace=traceback.format_exc(), log=ctx.logs[-30:])
        finish(report)
        return CHECK_FAILED
    report["seconds"]["parts"] = round(time.time() - t0, 2)
    report["parts"] = [part_stats(p, ctx.timings.get(p.name)) for p in parts]
    warns = BD.check_refs(parts)
    warns += BD.lint(parts)
    missing = BD.check_textures(parts, tex_dir)
    used = {f for p in parts for m in p.materials for f in (m.texture, m.toon, m.sphere) if f}
    unused = sorted(set(ctx.textures) - used)
    if unused:
        warns.append(f"textures written but no material uses them: {', '.join(unused[:8])}"
                     + (f" (+{len(unused) - 8} more)" if len(unused) > 8 else ""))
    report["warnings"] = ctx.warnings + warns
    report["log"] = ctx.logs if args.full else ctx.logs[-12:]
    report["textures"] = len(ctx.textures)
    if missing:
        report.update(ok=False, stage="parts", error=f"textures referenced but not written: {missing[:8]}")
        finish(report)
        return CHECK_FAILED
    if args.no_export:
        report["ok"] = True
        report["seconds"]["total"] = round(time.time() - t_all, 2)
        finish(report)
        return 0

    # ---- 2. assemble and write the PMX
    t0 = time.time()
    comment = cfg["comment"] or (f"{name}: original model built in code with mk model (mk-mmd). parts: "
                                 f"{', '.join(p.name for p in parts)}; spec {spec.digest()[:10]}")
    try:
        asm = AS.assemble(parts, name=name, scale=cfg["scale"], comment=comment)
        asm.pmx.validate()
    except (ValueError, KeyError) as e:
        report.update(ok=False, stage="assemble", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        finish(report)
        return CHECK_FAILED
    pmx_path = out / f"{name}.pmx"
    tmp_pmx = pmx_path.with_name(f".{pmx_path.name}.{os.getpid()}.tmp")
    pmx_io.write(asm.pmx, tmp_pmx)
    os.replace(tmp_pmx, pmx_path)
    keep = set(ctx.textures)
    for f in tex_dir.glob("*"):                      # textures of an earlier build that nothing writes any more
        if f.name not in keep:
            f.unlink(missing_ok=True)
    exp_prefix = out / ".mk" / "expected"
    exp_prefix.parent.mkdir(exist_ok=True)
    asm.save_expected(str(exp_prefix))
    report["seconds"]["assemble"] = round(time.time() - t0, 2)
    report["model"] = asm.stats
    report["pmx"] = str(pmx_path)
    report["pmx_mb"] = round(pmx_path.stat().st_size / 1e6, 2)
    report["warnings"] += asm.warnings

    # ---- 3. Blender: import back, verify, describe, review blend
    t0 = time.time()
    want_blend = not args.no_blend
    height = max((float(b.head[2]) for p in parts for b in p.bones), default=1.6)
    for p in parts:
        for m in p.meshes:
            if len(m.verts):
                height = max(height, float(np.asarray(m.verts)[:, 2].max()))
    blend_path = out / f"{name}.blend"
    try:
        res = run_blender("model_finish", {
            "pmx": str(pmx_path), "expected": str(exp_prefix), "blend": str(blend_path) if want_blend else None,
            "scale": cfg["scale"], "verify": not args.no_verify, "height": height}, timeout=1800)
    except bridge.BlenderError as e:
        report.update(ok=False, stage="blender", error=str(e).splitlines()[0], log_file=e.log)
        finish(report)
        return CHECK_FAILED
    report["seconds"]["blender"] = round(time.time() - t0, 2)
    ok = True
    if "verify" in res:
        v = res["verify"]
        report["verify"] = {"ok": not v["problems"], "problems": v["problems"], "numbers": v["numbers"]}
        ok = ok and not v["problems"]
    if "rig" in res:
        rig = res["rig"]
        rig_path = out / f"{name}.rig.json"
        tmp_rig = rig_path.with_name(f".{rig_path.name}.{os.getpid()}.tmp")
        tmp_rig.write_text(json.dumps(rig, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp_rig, rig_path)
        report["rig_json"] = str(rig_path)
        report["rig"] = rig_report(rig)
        if not report["rig"]["quirks"]:
            del report["rig"]["quirks"]
        ok = ok and not rig["missing_required"]
    if res.get("blend"):
        report["blend"] = res["blend"]
    report["ok"] = bool(ok)
    report["seconds"]["total"] = round(time.time() - t_all, 2)
    finish(report)
    return 0 if ok else CHECK_FAILED
