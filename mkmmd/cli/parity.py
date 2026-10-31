"""mk parity: how close mkr's frames are to Blender's (docs/design.md: Parity): compare two folders of frames, render a
project's frames with mkr against Blender's, the lab of small scenes, hill climbing over mkr's tunables."""
import argparse
import itertools
import sys
import time
from pathlib import Path

from .. import gpulock as GL
from .common import CHECK_FAILED, RUNTIME, UsageError, add_project_arg, emit, get_project, parse_frames

HELP = """How close mkr's frames are to Blender's (docs/design.md: Parity). Metrics per frame and pooled over all of them,
in 8-bit levels: mean |error|, signed bias per channel, % of pixels off by more than 8 and 32 levels (a pixel's error is
its largest channel's), p99 and max error, RMSE and PSNR; "1:1" is a pooled mean |error| <= 1.0 and <= 0.5 % of
pixels off by more than 8. Contact sheets show per frame Blender's frame, the test's and the error amplified as heat.

  compare REF_DIR TEST_DIR   any two folders of frames (matched by name: 00091.png is frame 91)
  frames                     a project's frames rendered by mkr from its bake against Blender's frames
                             (<project>/renders/<preset>/<output>, or --refs), skipping frames Blender draws in a
                             render-time look (silhouette, vector, reflection, a [[medium]] window) or through a fisheye
  lab [CASES]                the lab: small scenes that each isolate one thing (area lights, shadows, GGX, the
                             Principled BSDF, the view transform, the film filter, alpha, skinning, motion blur, depth
                             of field, an MMD model): --build makes them with Blender (scene, EEVEE frames, bake), then
                             mkr draws them; --variant renders a variant of them with Blender (noshadow, cycles,
                             cycles_noshadow, jittered), --against measures mkr against one, --separate splits the error
                             with them
  tune PARAMS.json           hill climbing over mkr's --params tunables on lab cases or a project's frames

mkr is $MK_MKR or <repo>/mkr/target/release/mkr. Output is JSON on stdout (lab and tune also print a table or their
steps on stderr).

Examples:
  mk parity compare renders/draft/16x9 .mk/parity/run1/16x9/frames --signed
  mk parity frames --output 16x9 --preset draft --frames 100:400:30
  mk parity lab --build --source ~/Videos/clip --cast rin       # build every case, then measure mkr on them
  mk parity lab 'area_*' ggx --per-frame
  mk parity lab --variant cycles 'area_*'                     # Cycles' direct light, the truth EEVEE approximates
  mk parity lab 'area_*' --against cycles
  mk parity lab --separate --per-frame penumbra contact_shadow
  mk parity tune tune.json --cases 'area_*' point spot --iterations 80
"""

TUNE_HELP = """Hill-climb mkr's tunables (its --params file) on the mean over every frame of one metric: of the lab's
--cases, of a --project's frames, or of both (then the project's mean plus the cases' mean, so neither side's frame
count outweighs the other, each times its "weights" entry, 1 unless named).

PARAMS.json:
  {"params": {"<name>": {"start": 1.0, "step": 0.05, "min": 0.5, "max": 2.0},     a number (start: mkr's default
              "<name>": {"start": 6, "step": 2, "min": 1, "int": true},            when left out: `mkr params`)
              "<name>": {"choices": ["gauss", "box"]}},                            a choice (coord only)
   "fixed": {"<name>": value},                       passed to mkr unchanged
   "metric": "mean_abs", "method": "coord", "iterations": 60, "min_step": 0.001, "pool": false,
   "weights": {"project": 1, "lab": 1}}             with both: how much each side's mean counts

metric: mean_abs, rmse, over8, over32, p99, max, bias_abs (mean |bias| of the channels) or psnr (maximised).
pool: score each output's (or case's) frames pooled over their pixels, as `compare` reports them, and take the mean
over the outputs (mean_abs, over8, over32, bias_abs); else the mean over every frame.
method: coord (coordinate descent: a step up and down per parameter, going on while it pays, halving the steps when a
sweep moves nothing) or nelder-mead (scipy, numbers only). Every evaluation is a line of OUT/tune.jsonl; the best so
far is OUT/best.json and OUT/best_params.json (mkr's --params file); at the end the best is drawn once more with
contact sheets under OUT/scratch/ (--no-sheets: not). OUT defaults to <lab>/_tune/<time>/ or
<project>/.mk/parity/tune-<time>/.

--resume LOG goes on from an earlier run's OUT/tune.jsonl (same file, options and mkr), one stopped or cut short: the
search walks through its points again at no cost and --iterations counts the new evaluations. Each evaluation (its
renders and their comparisons) holds the GPU lock, released between evaluations: one wait each."""


def add(sub):
    p = sub.add_parser("parity", help="mkr's frames against Blender's: metrics, sheets, the lab, hill climbing",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ps = p.add_subparsers(dest="parity", metavar="SUBCOMMAND", required=True)

    c = ps.add_parser("compare", help="metrics and sheets of two folders of frames",
                      description="Compare the frames of TEST_DIR with those of REF_DIR (matched by file name).")
    c.add_argument("ref", metavar="REF_DIR", help="the reference frames (Blender's)")
    c.add_argument("test", metavar="TEST_DIR", help="the frames to measure (mkr's)")
    c.add_argument("--frames", help="frame spec: only these frames (default: every frame both folders have)")
    c.add_argument("--out", help="where summary.json and the sheets go (default: TEST_DIR-compare beside TEST_DIR)")
    _measure_args(c)
    c.set_defaults(func=run_compare)

    f = ps.add_parser("frames", help="a project's frames rendered by mkr against Blender's",
                      description="Render frames with mkr from the project's bake into <project>/.mk/parity/<run>/"
                                  "<output>/frames/ and compare them with Blender's (compare/ beside them). Frames "
                                  "Blender draws in a render-time look (a shot's or a [[medium]] window's) or through a "
                                  "fisheye camera are left out and listed with the reason.")
    f.add_argument("--output", action="append", metavar="NAME", help="output(s) (default: every output)")
    f.add_argument("--frames", help="frame spec (default: every frame of the clip the references have)")
    f.add_argument("--preset", default="final", choices=("draft", "preview", "final"),
                   help="whose frames: <project>/renders/<preset>/<output> (default final); also motion blur on or off")
    f.add_argument("--refs", metavar="DIR", help="Blender's frames: a folder of frames, or of <output>/ folders")
    f.add_argument("--run", metavar="NAME", help="the run's folder name (default: the time)")
    _mkr_args(f)
    _measure_args(f)
    add_project_arg(f)
    f.add_argument("--mkr-args", nargs=argparse.REMAINDER, default=[], help="the rest goes to `mkr render` (last)")
    f.set_defaults(func=run_frames)

    lab = ps.add_parser("lab", help="the lab: small scenes, Blender's frames of them, mkr's",
                        description="Measure mkr on the lab's cases (mkmmd/render/lab.py), each a tiny mk project in "
                                    "<lab>/<case>/ (--root, $MK_LAB, else ~/.cache/mk/lab). --build makes the missing "
                                    "ones with Blender: the scene, EEVEE's frames at 64 samples (8-bit, 16-bit, "
                                    "scene-linear EXR) and the bake. mmd_model and mmd_sphere show a cast member of a "
                                    "built project at rest (--source, --cast). --variant renders Blender's frames of a "
                                    "variant of each built case from its saved scene into <case>/variants/<name>/ "
                                    "(noshadow: EEVEE with every shadow off; cycles, cycles_noshadow: Cycles, direct "
                                    "light only; jittered: EEVEE's depth of field by camera jitter); --against measures "
                                    "mkr against a variant instead of EEVEE's frames; "
                                    "--separate splits each case's error: EEVEE and mkr against Cycles with and "
                                    "without shadows, and their shadow terms against each other.")
    lab.add_argument("cases", nargs="*", metavar="CASE", help="case names or patterns (default: every case)")
    lab.add_argument("--root", metavar="DIR", help="the lab folder")
    lab.add_argument("--list", action="store_true", help="list the cases and whether they are built")
    lab.add_argument("--build", action="store_true", help="build the cases not built yet (Blender)")
    lab.add_argument("--rebuild", action="store_true", help="build the cases again")
    lab.add_argument("--rebake", action="store_true", help="bake the built cases again (`mk build --bake` on each)")
    lab.add_argument("--refs-only", action="store_true", help="stop after building: no mkr")
    lab.add_argument("--source", metavar="DIR|BLEND", help="the built project (or .blend) the mmd cases take a member from")
    lab.add_argument("--cast", metavar="NAME", help="that member ([[cast]] name; with a .blend its armature)")
    lab.add_argument("--variant", metavar="NAME", help="render this variant of the built cases with Blender (again "
                                                       "when there): noshadow, cycles, cycles_noshadow or jittered; "
                                                       "no mkr")
    lab.add_argument("--variant-samples", type=int, metavar="N", help="with --variant: its samples")
    lab.add_argument("--seed", type=int, help="with --variant: Cycles' seed (default 0)")
    lab.add_argument("--against", metavar="NAME", help="measure mkr against this variant's frames, not EEVEE's")
    lab.add_argument("--separate", action="store_true",
                     help="split each case's error: mkr drawn as it is and with every shadow off, measured against "
                          "EEVEE's frames and the variants (--variant renders those)")
    lab.add_argument("--probes", metavar="FILE", help="with --separate: {case: {lines, boxes}} instead of the lab's own")
    lab.add_argument("--params", metavar="FILE", help="mkr's tunables (--params)")
    lab.add_argument("--per-frame", action="store_true", help="the table has a row per frame too")
    _measure_args(lab, region=False)
    lab.add_argument("--mkr-args", nargs=argparse.REMAINDER, default=[], help="the rest goes to `mkr render` (last)")
    lab.set_defaults(func=run_lab)

    t = ps.add_parser("tune", help="hill climbing over mkr's tunables", description=TUNE_HELP,
                      formatter_class=argparse.RawDescriptionHelpFormatter)
    t.add_argument("spec", metavar="PARAMS.json", help="what to tune (see above)")
    t.add_argument("--cases", nargs="+", metavar="CASE", help="on these lab cases (names or patterns)")
    t.add_argument("--project", metavar="DIR", help="on this project's frames (see `mk parity frames`); with --cases "
                                                    "too, the score is the project's mean plus the cases' mean")
    t.add_argument("--root", metavar="DIR", help="the lab folder (with --cases)")
    t.add_argument("--output", action="append", metavar="NAME", help="with --project: output(s) (default: every one)")
    t.add_argument("--frames", help="with --project: frame spec (default: every frame the references have)")
    t.add_argument("--preset", default="final", choices=("draft", "preview", "final"), help="with --project: as for frames")
    t.add_argument("--refs", metavar="DIR", help="with --project: Blender's frames")
    t.add_argument("--metric", choices=("mean_abs", "rmse", "over8", "over32", "p99", "max", "bias_abs", "psnr"),
                   help="what to minimise (default: the file's, else mean_abs)")
    t.add_argument("--method", choices=("coord", "nelder-mead"), help="default: the file's, else coord")
    t.add_argument("--iterations", type=int, help="evaluations at most (default: the file's, else 60)")
    t.add_argument("--pool", action=argparse.BooleanOptionalAction, default=None,
                   help="score each output's (case's) frames pooled over their pixels, as compare reports them, then "
                        "the mean over the outputs (default: the file's \"pool\", else the mean over every frame)")
    t.add_argument("--out", metavar="DIR", help="the log, the best params and the sheets")
    t.add_argument("--resume", metavar="LOG", help="go on from an earlier run's OUT/tune.jsonl (see above)")
    t.add_argument("--sheets", action=argparse.BooleanOptionalAction, default=True,
                   help="draw the best once more with contact sheets at the end (default: yes)")
    t.add_argument("--figure", type=float, nargs="?", const=4.0, metavar="LEVELS",
                   help="score only the figure: pixels where either frame has a channel above LEVELS (default 4)")
    _mkr_args(t, params=False)
    t.add_argument("--mkr-args", nargs=argparse.REMAINDER, default=[], help="the rest goes to `mkr render` (last)")
    t.set_defaults(func=run_tune)


def _measure_args(p, region=True):
    p.add_argument("--mask", metavar="FILE|DIR", help="measure only where this image is white (a folder: one per frame)")
    if region:
        p.add_argument("--region", metavar="X0,Y0,X1,Y1", help="measure only inside this box (fractions of the frame "
                                                              "when all <= 1, else pixels)")
    p.add_argument("--figure", type=float, nargs="?", const=4.0, metavar="LEVELS",
                   help="measure only the figure: pixels where either frame has a channel above LEVELS (default 4), "
                        "the black void left out")
    p.add_argument("--gain", type=float, default=8.0, help="the heat's amplification (default 8: 32 levels saturate)")
    p.add_argument("--signed", action="store_true", help="a fourth panel: the signed luma error (red: brighter)")
    p.add_argument("--no-sheet", action="store_true", help="no contact sheets")
    p.add_argument("--gate", action="store_true", help="exit 1 unless the pooled metrics are 1:1")


def _mkr_args(p, params=True):
    p.add_argument("--samples", type=int, help="mkr's samples (default 64; with --preset final, mkr's final sampling: "
                   "36 samples of twice the scene's shadow rays, as mk render draws them)")
    p.add_argument("--percent", type=int, help="resolution percentage (default: the references')")
    p.add_argument("--motion-blur", action=argparse.BooleanOptionalAction, default=None,
                   help="mkr's motion blur (default: the preset's: on for preview and final)")
    p.add_argument("--shutter", type=float, help="the motion blur shutter (default: [render] shutter, else 0.35)")
    if params:
        p.add_argument("--params", metavar="FILE", help="mkr's tunables (--params)")


def _region(args):
    r = getattr(args, "region", None)
    if r is None:
        return None
    try:
        box = [float(v) for v in r.split(",")]
    except ValueError:
        box = []
    if len(box) != 4:
        raise UsageError(f"--region {r!r}: four numbers X0,Y0,X1,Y1")
    return box


def _measure_kw(args):
    return {"mask": args.mask, "region": _region(args), "figure": args.figure, "gain": args.gain, "signed": args.signed}


def _gated(args, ok):
    return CHECK_FAILED if args.gate and not ok else 0


# ================================================================================================= compare
def run_compare(args):
    from ..render import parity as PY
    frames = parse_frames(args.frames, None) if args.frames else None
    test = Path(args.test).expanduser().resolve()
    out = Path(args.out).expanduser() if args.out else test.with_name(test.name + "-compare")
    try:
        s = PY.compare(Path(args.ref).expanduser(), test, out, frames=frames, draw=not args.no_sheet, **_measure_kw(args))
    except PY.ParityError as e:
        raise UsageError(str(e)) from None
    if not s["frames"] and not s["errors"]:
        raise UsageError(f"no frame is in both {args.ref} and {args.test}" + (" (of --frames)" if frames else ""))
    emit(PY.brief(s))
    return _gated(args, s["pass"])


# ================================================================================================= frames
def _mkr_cfg(args, proj, preset):
    from ..render import mkr as MKR
    from .render import PRESETS
    mb = args.motion_blur if args.motion_blur is not None else PRESETS[preset]["motion_blur"]
    shutter = args.shutter or float(proj.data.get("render", {}).get("shutter", 0.35))
    return {"samples": args.samples or 64, "motion_blur": bool(mb), "shutter": shutter,
            "sampling": MKR.sampling(preset, proj.data.get("render"), args.samples)}


def _plans(args, proj):
    from ..render import parity as PY
    outs = args.output or [o.name for o in proj.outputs]
    if not outs:
        raise UsageError("no [[output]] in mk.toml")
    frames = parse_frames(args.frames, proj) if args.frames else None
    try:
        bake, plans = PY.plan(proj, outs, frames, args.refs, args.preset)
    except PY.ParityError as e:
        raise UsageError(str(e)) from None
    if args.percent:
        for p in plans.values():
            p["percent"] = args.percent
    if not any(p["frames"] for p in plans.values()):
        raise UsageError("no lit frame with a reference to compare: " +
                         "; ".join(f"{o}: {p['ref']}" for o, p in plans.items()))
    return bake, plans


def _plan_report(p):
    from ..render import mkr as MKR
    return {"refs": str(p["ref"]), "drawn": MKR.spec(p["frames"]), "percent": p["percent"], "skipped": p["skipped"],
            **({"missing": MKR.spec(p["missing"])} if p["missing"] else {}),
            **({"outside": MKR.spec(p["outside"])} if p["outside"] else {})}


def _need_mkr():
    """mkr's binary, or a UsageError that says how to get it."""
    from ..render import mkr as MKR
    try:
        return MKR.binary()
    except MKR.EngineError as e:
        raise UsageError(str(e)) from None


def run_frames(args):
    import json

    from ..core import jsonx
    from ..render import mkr as MKR
    from ..render import parity as PY
    proj = get_project(args, required=True)
    bake, plans = _plans(args, proj)
    _need_mkr()
    cfg = _mkr_cfg(args, proj, args.preset)
    run_dir = proj.mk_dir / "parity" / (args.run or time.strftime("%Y%m%d-%H%M%S"))
    extra = (["--params", str(Path(args.params).expanduser().resolve())] if args.params else []) + list(args.mkr_args)
    report = {"run": str(run_dir), "bake": str(bake), "mkr": dict(cfg, extra=extra),
              "outputs": {o: _plan_report(p) for o, p in plans.items()}}
    t0 = time.time()
    try:
        res = PY.render_compare(bake, plans, run_dir, cfg, extra, draw=not args.no_sheet, **_measure_kw(args))
    except MKR.EngineError as e:
        emit(dict(report, error=f"mkr: {e}"))
        return RUNTIME
    except PY.ParityError as e:
        raise UsageError(str(e)) from None
    for o, r in res.items():
        report["outputs"][o].update(mkr=r["mkr"], **PY.brief(r["summary"]))
    report["overall"] = PY.pool([r["summary"] for r in res.values()])
    report["pass"] = PY.passes(report["overall"])
    report["seconds"] = round(time.time() - t0, 1)
    (run_dir / "run.json").write_text(json.dumps(jsonx.to_jsonable(report, 6), indent=1), encoding="utf-8")
    emit(report)
    return _gated(args, report["pass"])


# ================================================================================================= lab
def run_lab(args):
    from ..render import lab as LB
    from ..render import mkr as MKR
    from ..render import parity as PY
    try:
        lab = LB.root(args.root)
        names = LB.names(args.cases)
    except LB.LabError as e:
        raise UsageError(str(e)) from None
    if args.list:
        emit({"lab": str(lab), "cases": {n: {"doc": LB.CASES[n]["doc"], "built": LB.is_built(lab / n),
                                             **({"source": True} if LB.CASES[n].get("source") else {})} for n in names}})
        return 0
    if args.source and not any(LB.CASES[n].get("source") for n in names):
        raise UsageError("--source is for the mmd cases only")
    for flag, name in (("--variant", args.variant), ("--against", args.against)):
        if name is not None and name not in LB.VARIANTS:
            raise UsageError(f"{flag} {name!r}: no such variant (variants: {', '.join(LB.VARIANTS)})")
    if args.variant and (args.against or args.separate):
        raise UsageError("--variant only renders Blender's frames: measure with --against or --separate after it")
    if args.against and args.separate:
        raise UsageError("--against and --separate measure differently: one at a time")
    if (args.variant_samples is not None or args.seed is not None) and not args.variant:
        raise UsageError("--variant-samples and --seed go with --variant")
    if args.probes and not args.separate:
        raise UsageError("--probes goes with --separate")
    built, notes = {}, {}
    for n in names:
        d = lab / n
        if args.rebuild or (args.build and not LB.is_built(d)):
            src, cast = args.source, args.cast
            if LB.CASES[n].get("source") and src is None and LB.is_built(d):
                prev = LB.load_case(d).get("source") or {}
                src, cast = prev.get("blend"), prev.get("armature")
            if LB.CASES[n].get("source") and src is None:
                notes[n] = "not built: it shows a cast member of a built project (--source DIR --cast NAME)"
                continue
            print(f"lab: building {n}", file=sys.stderr, flush=True)
            try:
                rec = LB.build(n, lab, src, cast)
            except LB.LabError as e:
                raise UsageError(f"{n}: {e}") from None
            bake = rec["report"]["bake"]
            built[n] = {"seconds": rec["seconds"], "frames": rec["report"]["refs"]["frames"],
                        "bake": {"version": bake.get("version")} if "error" not in bake else {"error": bake["error"]}}
        elif args.rebake and LB.is_built(d):
            print(f"lab: baking {n}", file=sys.stderr, flush=True)
            bake = LB.rebake(n, lab)
            built[n] = {"rebaked": {"version": bake.get("version")} if "error" not in bake else {"error": bake["error"]}}
    out = {"lab": str(lab)}
    if built:
        out["built"] = built
    if args.refs_only:
        emit(dict(out, notes=notes) if notes else out)
        return 0
    if args.variant:
        return _lab_variants(args, lab, names, out, notes)
    _need_mkr()
    extra = (["--params", str(Path(args.params).expanduser().resolve())] if args.params else []) + list(args.mkr_args)
    if args.separate:
        return _lab_separate(args, lab, names, out, notes, extra)
    rows, cases, ok = [], {}, True
    for n in names:
        d = lab / n
        if not LB.is_built(d):
            notes.setdefault(n, "not built: `mk parity lab --build`")
            continue
        rec = LB.load_case(d)
        if "error" in (rec["report"].get("bake") or {}):
            notes[n] = f"its bake failed: {rec['report']['bake']['error']} (`mk parity lab --rebake {n}`)"
            continue
        ref = LB.variant_dir(d, args.against) if args.against else None
        if ref is not None and not LB.has_variant(d, args.against):
            notes[n] = f"no {args.against} variant: `mk parity lab --variant {args.against} {n}`"
            continue
        try:
            reps = LB.render(n, lab, extra=extra)
        except MKR.EngineError as e:
            cases[n] = {"error": f"mkr: {e}"}
            ok = False
            continue
        sums = LB.measure(n, lab, ref=ref, out=ref / "compare" if ref is not None else None, draw=not args.no_sheet,
                          mask=args.mask, figure=args.figure, gain=args.gain, signed=args.signed)
        rows += LB.rows(n, rec, sums, {o: r.get("seconds") for o, r in reps.items()})
        cases[n] = {o: {"mkr": reps[o], **PY.brief(s)} for o, s in sums.items()}
        ok = ok and all(s["pass"] for s in sums.values())
    if rows:
        print(LB.table(rows, args.per_frame), file=sys.stderr, flush=True)
    emit(dict(out, **({"against": args.against} if args.against else {}), cases=cases, rows=rows,
              **({"notes": notes} if notes else {})))
    if any("error" in c for c in cases.values()):
        return RUNTIME
    return _gated(args, ok and bool(rows))


def _lab_variants(args, lab, names, out, notes):
    """`mk parity lab --variant NAME`: Blender's frames of that variant of every named built case."""
    from .. import bridge
    from ..render import lab as LB
    done, failed = {}, {}
    for n in names:
        if not LB.is_built(lab / n):
            notes.setdefault(n, "not built: `mk parity lab --build`")
            continue
        print(f"lab: rendering {n} as {args.variant}", file=sys.stderr, flush=True)
        try:
            rec = LB.render_variant(n, lab, args.variant, samples=args.variant_samples, seed=args.seed)
        except bridge.BlenderError as e:
            failed[n] = {"error": str(e), **({"log": e.log} if e.log else {})}
            continue
        rep = rec["report"]
        done[n] = {"frames": rep["frames"], "seconds": rec["seconds"], "s_per_frame": rep["s_per_frame"],
                   "engine": rep["engine"], **({"device": rep["device"]} if "device" in rep else {})}
    emit(dict(out, variant=args.variant, rendered=done, **({"failed": failed} if failed else {}),
              **({"notes": notes} if notes else {})))
    return RUNTIME if failed else 0


def _lab_separate(args, lab, names, out, notes, extra):
    """`mk parity lab --separate`: mkr drawn as it is and with {"shadows": false}, each case's error split (lab.separate),
    the table and the probes on stderr."""
    import json
    import shutil
    import tempfile

    from ..render import lab as LB
    from ..render import mkr as MKR
    from ..render import parity as PY

    def load(path, what):
        try:
            return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise UsageError(f"{what} {path}: {e}") from None
    probes = load(args.probes, "--probes") if args.probes else {}
    params = load(args.params, "--params") if args.params else {}
    try:
        shadows = "shadows" in PY.mkr_params()
    except (MKR.EngineError, PY.ParityError, OSError) as e:
        raise UsageError(f"`mkr params`: {e}") from None
    if not shadows:
        notes["mkr"] = "mkr has no `shadows` tunable (`mkr params`): nothing is measured without mkr's shadows"
    rows, cases = [], {}
    with tempfile.TemporaryDirectory(prefix="mk-separate-") as tmp:
        noshadow = Path(tmp) / "noshadow.json"
        noshadow.write_text(json.dumps(dict(params, shadows=False)), encoding="utf-8")
        for n in names:
            d = lab / n
            if not LB.is_built(d):
                notes.setdefault(n, "not built: `mk parity lab --build`")
                continue
            rec = LB.load_case(d)
            if "error" in (rec["report"].get("bake") or {}):
                notes[n] = f"its bake failed: {rec['report']['bake']['error']} (`mk parity lab --rebake {n}`)"
                continue
            print(f"lab: separating {n}", file=sys.stderr, flush=True)
            try:
                LB.render(n, lab, extra=extra)
                if shadows:
                    LB.render(n, lab, out=d / "mkr_noshadow", extra=["--params", str(noshadow), *args.mkr_args])
                else:
                    shutil.rmtree(d / "mkr_noshadow", ignore_errors=True)
            except MKR.EngineError as e:
                cases[n] = {"error": f"mkr: {e}"}
                continue
            try:
                sep = LB.separate(n, lab, draw=not args.no_sheet, figure=args.figure, probes=probes.get(n))
            except LB.LabError as e:
                raise UsageError(f"{n}: {e}") from None
            rows += LB.separation_rows(n, rec, sep)
            cases[n] = LB.separation_brief(sep)
            lacking = sorted({x.split("/")[1] for v in sep["missing"].values() for x in v if x.startswith("variants/")})
            if lacking:
                notes[n] = f"no {', '.join(lacking)} variant: `mk parity lab --variant NAME {n}`"
    if rows:
        print(LB.separation_table(rows, args.per_frame), file=sys.stderr, flush=True)
        lines = [line for n, c in cases.items() if "probes" in c for line in LB.probe_table(n, c["probes"])]
        if lines:
            print("\n".join(lines), file=sys.stderr, flush=True)
    emit(dict(out, cases=cases, rows=rows, **({"notes": notes} if notes else {})))
    return RUNTIME if any("error" in c for c in cases.values()) else 0


# ================================================================================================= tune
def run_tune(args):
    from ..render import lab as LB
    from ..render import mkr as MKR
    from ..render import parity as PY
    _need_mkr()
    try:
        tspec = PY.load_space(args.spec)
    except PY.ParityError as e:
        raise UsageError(str(e)) from None
    defaults = None
    if any(p["start"] is None for p in tspec["params"].values()):
        try:
            defaults = PY.mkr_params()
        except (MKR.EngineError, PY.ParityError, OSError) as e:
            raise UsageError(f"a param without a start needs mkr's defaults: {e}") from None
    if not (args.cases or args.project):
        raise UsageError("tune needs --cases, --project or both")
    try:
        resume = PY.read_log(Path(args.resume).expanduser()) if args.resume else []
    except PY.ParityError as e:
        raise UsageError(f"--resume: {e}") from None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    extra = list(args.mkr_args)
    both = bool(args.cases and args.project)
    evaluators, on, out = {}, {}, None
    if args.project:
        args.project = str(Path(args.project).expanduser())
        proj = get_project(args, required=True)
        bake, plans = _plans(args, proj)
        out = Path(args.out).expanduser() if args.out else proj.mk_dir / "parity" / f"tune-{stamp}"
        evaluators["project"] = PY.project_evaluator(bake, plans, out / "scratch" / ("project" if both else ""),
                                                     _mkr_cfg(args, proj, args.preset), extra, figure=args.figure)
        on.update(project=str(proj.root), outputs={o: _plan_report(p) for o, p in plans.items()})
    if args.cases:
        try:
            lab = LB.root(args.root)
            names = LB.names(args.cases)
        except LB.LabError as e:
            raise UsageError(str(e)) from None
        lacking = [n for n in names if not LB.is_built(lab / n)]
        if lacking:
            raise UsageError(f"not built: {', '.join(lacking)} (`mk parity lab --build`)")
        out = out or (Path(args.out).expanduser() if args.out else lab / "_tune" / stamp)
        evaluators["lab"] = LB.evaluator(names, lab, out / "scratch" / ("lab" if both else ""), extra, figure=args.figure)
        on.update(cases=names, lab=str(lab))
    # with both, each frame's part is named "project:<output>" or "lab:<case>/<output>" and the score sums the two means
    evaluate = _turn_each(PY.combined(evaluators) if both else next(iter(evaluators.values())))
    try:
        res = PY.tune(evaluate, tspec, out, say=lambda line: print(line, file=sys.stderr, flush=True), defaults=defaults,
                      method=args.method, iterations=args.iterations, metric=args.metric, pool=args.pool,
                      resume=resume)
        if args.sheets:
            evaluate(Path(res["best_params"]), draw=True)
    except PY.ParityError as e:
        emit({"on": on, "out": str(out), "error": str(e)})
        return RUNTIME
    except MKR.EngineError as e:
        emit({"on": on, "out": str(out), "error": f"mkr: {e}"})
        return RUNTIME
    emit(dict(res, on=on, out=str(out), **({"sheets": str(out / "scratch")} if args.sheets else {})))
    return 0


def _turn_each(evaluate):
    """`evaluate` holding the GPU lock for each evaluation, its renders and their comparisons, and free between them: a
    tune waits once per evaluation rather than once per render."""
    count = itertools.count(1)

    def held(params_file, draw=False):
        with GL.held(f"mk parity tune: eval {next(count)}"):
            return evaluate(params_file, draw=draw)
    return held
