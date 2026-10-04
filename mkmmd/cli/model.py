"""mk model: build an original character model from a spec (part builders in code -> PMX -> rig.json -> review .blend)."""
import argparse
import json
import os
import random
import re
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
described like `mk inspect` does (rig.json). A review .blend with a neutral studio works with `mk look`.

Examples:
  mk model build ~/Projects/mk-tests/rin_model/model.toml
  mk model build model.toml --only head                  # head + the parts it needs, into <out>/only_head/
  mk model build model.toml --no-export                  # run and check the builders only (no Blender)
  mk model build model.toml --set hair.length=0.3 --out /tmp/try
  mk look <out>/<name>.blend --view front,3q --target "bone('head').head" --dist 1.2 --frames 1
  mk model info model.toml                               # what would be built, in which order

Output files in <out>: <name>.pmx, tex/*.png, <name>.blend, <name>.rig.json, build.json (the printed report).
"""


def add(sub):
    p = sub.add_parser("model", help="build an original character (parts -> PMX, rig.json, review .blend)",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ss = p.add_subparsers(dest="model_command", metavar="ACTION", required=True)
    b = ss.add_parser("build", help="build the parts, export the PMX, import it back and verify it",
                      description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    b.add_argument("spec", metavar="SPEC.toml")
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
    i.add_argument("spec", metavar="SPEC.toml")
    i.add_argument("--only", metavar="PARTS")
    i.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides")
    i.set_defaults(func=run_info)
    s = ss.add_parser("studio", help="copy a built scene with the neutral review studio (grey world, floor, lights)")
    s.add_argument("scene", metavar="SCENE.blend")
    s.add_argument("--out", required=True, metavar="OUT.blend")
    s.add_argument("--floor", action="append", default=[], metavar="X,Y", help="a floor disc at x,y (repeatable)")
    s.add_argument("--lights", action="store_true", help="add key/fill/rim suns (default: the scene has its own)")
    s.add_argument("--height", type=float, default=1.6, help="model height for aiming the lights")
    s.set_defaults(func=run_studio)


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
