"""mkr, mk-mmd's own renderer of the lit scene (docs/design.md: Rendering): `[render] engine = "mkr"`. It draws the cut's
frames from the scene's bake (version 5: `mk build` writes it) on the GPU, as Blender's EEVEE draws them, and the layers
`mk render` writes beside them: plates and bare frames (lit frames through another shot's camera, or with a glitch's
objects hidden), figures (a collage's coverage, as Workbench antialiases it), points, keep masks and screen type. A frame
or layer in a render-time look (a shot's or a [[medium]] window's silhouette or vector look) is drawn as passes, the
ones styles.Looks renders in Blender (mkr/mkr/src/looks.rs), which worker processes here compose with the same
compose_silhouette / compose_vector while mkr draws the next (`Composer`, mkmmd.render.compose.finish_look).

    req = request(doc, demands, keep, frames, outputs)   # every layer and look, patterns resolved on the bake
    pl = plan(bake, outputs, frames, req)                 # `mkr info`: which frames and layers mkr draws, why Blender the rest
    mine, theirs = split(req, pl["outputs"][name], frames_dir, name)
    draw(bake, name, pl["outputs"][name]["draw"], frames_dir, cfg, mine, looks=looks(doc, blob, req, name, size, cfg))

A frame mkr cannot draw (a reflection, a fisheye camera, something it cannot hold) and a layer it cannot draw stay
Blender's: `split` hands them to its job (ops_render's `demands`, `mk_keep`, `mk_screen`)."""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import memory as MEM
from ..core import keep as KP
from ..core import medium as MD
from ..core import rings as RG
from ..core import screentype as SRT
from ..core import shotstyle as SS
from ..core import transition as TR
from . import compose as CO
from . import plan as PL

REPO = Path(__file__).resolve().parents[2]
LOOK_KINDS = ("silhouette", "vector")         # the looks mkr draws the passes of (a reflection is Blender's)
LOOK_ROLES = ("plate", "bare", "matte")       # the layers drawn in their shot's look (a figure is coverage alone)
SHM = "/dev/shm"
COMPOSE_JOBS = 4                             # compose workers at most, as mk post's (mkmmd.postframes.MAX_JOBS)
COMPOSE_PEAK = 1.2e9                         # bytes a compose worker peaks at (1.1 GB measured on 1080 vector frames)
MEMORY_SHARE = 0.5                           # of the memory available, what the workers may take
# How mkr draws `mk render --preset final` (docs/design.md: mkr's samples): EEVEE's 64 samples as 36, each with twice the
# scene's shadow rays, from Sobol's sequence, and 8 motion blur taps. The shadows' noise is most of what fewer samples
# leave; this keeps it, as close to EEVEE's frames as mkr's 64 samples, for a fifth to a third less GPU time.
FINAL = {"samples": 36, "params": {"shadow_rays": 2, "shadow_sequence": "sobol", "mb_taps": 8}}


def sampling(preset, render=None, samples=None):
    """mkr's own sampling for `preset` (FINAL for final) when the preset's own samples apply: none asked for (a
    command's --samples) and none in the project's [render] table `render`; else None."""
    return FINAL if preset == "final" and not samples and "samples" not in (render or {}) else None


class EngineError(RuntimeError):
    pass


def binary():
    """The mkr executable: $MK_MKR, else the release build in the repository's mkr/ workspace."""
    given = os.environ.get("MK_MKR")
    path = Path(given).expanduser() if given else REPO / "mkr" / "target" / "release" / "mkr"
    if not path.is_file():
        raise EngineError(f"mkr not found at {path}: build it (`cargo build --release --manifest-path "
                          f"{REPO / 'mkr' / 'Cargo.toml'}`) or set MK_MKR")
    return path


def spec(frames):
    """Frame numbers as mkr's frame spec: runs of consecutive frames as `a:b`, joined by commas."""
    out, run = [], None
    for f in sorted({int(f) for f in frames}):
        if run is not None and f == run[1] + 1:
            run[1] = f
            continue
        if run is not None:
            out.append(f"{run[0]}:{run[1]}" if run[1] > run[0] else str(run[0]))
        run = [f, f]
    if run is not None:
        out.append(f"{run[0]}:{run[1]}" if run[1] > run[0] else str(run[0]))
    return ",".join(out)


def _call(argv, timeout, layers=None, composer=None, params=None):
    """Run mkr with `argv` (and `--layers FILE` holding `layers`, a request, and `--params FILE` holding `params`, its
    tunables, when given); its JSON report. With a `composer` (its looks' finish) mkr is stopped as soon as composing
    fails."""
    paths = []
    try:
        for flag, doc in (("--layers", layers), ("--params", params)):
            if doc is None:
                continue
            kind = "mkr_" + flag[2:]
            MEM.sweep(kind, [tempfile.gettempdir()])               # what runs killed outright left in memory
            fd, path = tempfile.mkstemp(prefix=MEM.scratch_prefix(kind), suffix=".json")
            paths.append(path)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, ensure_ascii=False)
            argv = [*argv, flag, path]
        proc = subprocess.Popen([str(binary()), *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        t0 = time.time()
        while True:
            try:
                out, err = proc.communicate(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if time.time() - t0 > timeout or (composer is not None and composer.failed()):
                    proc.kill()
                    proc.communicate()
                    if composer is not None and composer.failed():
                        raise EngineError(f"composing mkr's look passes: {composer.errors[0]}") from None
                    raise EngineError(f"mkr {argv[0]} took longer than {timeout} s") from None
    finally:
        for path in paths:
            os.unlink(path)
    if proc.returncode != 0:
        tail = (err or out).strip().splitlines()[-12:]
        raise EngineError(f"mkr {argv[0]} failed ({proc.returncode}): " + "\n".join(tail))
    try:
        return json.loads(out)
    except ValueError:
        raise EngineError(f"mkr {argv[0]} printed no JSON report: {out[-400:]!r}") from None


# ================================================================================================= the request
def records(doc):
    """What object patterns match (mkmmd.core.shotstyle.Obj) for every object the bake knows: bake.json `objects` (what
    a look sorts) and the lit section's `records` (the others its meshes name)."""
    out = [SS.Obj(o["name"], tuple(o.get("collections") or ()), frozenset(o.get("props") or ())) for o in doc.get("objects") or []]
    for name, r in sorted(((doc.get("lit") or {}).get("records") or {}).items()):
        out.append(SS.Obj(name, tuple(r.get("collections") or ()), frozenset(r.get("props") or ())))
    return out


def keep_materials(doc, keep):
    """The bake's materials `[post] keep` names ({armature, materials}: PMX names) on its member's meshes, as
    ops_render.kept_materials finds them in the scene; a name the model does not have is a ValueError listing the ones
    it has."""
    lit = doc.get("lit") or {}
    mats = lit.get("materials") or {}
    have = {}
    for m in lit.get("meshes") or []:
        if m.get("armature") != keep["armature"]:
            continue
        for name in m.get("materials") or []:
            if name is None or name not in mats:
                continue
            pmx = ((mats[name].get("settings") or {}).get("mmd_material") or {}).get("name_j") or name
            have.setdefault(pmx, set()).add(name)
    if not have:
        raise ValueError(f"[post] keep: the bake has no mesh of {keep['armature']}")
    lacking = [n for n in keep["materials"] if n not in have]
    if lacking:
        raise ValueError(f"[post] keep: {keep['armature']} has no material {', '.join(lacking)} (it has "
                         f"{', '.join(sorted(have))})")
    return sorted(set().union(*(have[n] for n in keep["materials"])))


def request(doc, demands, keep, frames, outputs=()):
    """Every layer of `frames` for mkr to sort (`plan`) or draw (`draw`): {"items": {frame: [item]} (TR.demands items, a
    figure's and a bare frame's object patterns resolved on the bake as `names`), "keep": {"materials": [bake material
    names], "frames": [...]} or None, "screen": [frames] (those with screen type get its layer) or None}, and with
    `outputs` the looks they are drawn in ("looks": {"outputs": {output: `look_jobs`}}; an item seen through a look
    gets "looks" {output: job id}, a figure seen through one "look_names" {output: its names less what the look
    hides})."""
    recs = records(doc)
    items = {}
    for f, its in demands.items():
        out = []
        for it in its:
            it = dict(it)
            if it["kind"] == "figure":
                it["names"] = SS.select(recs, it["objects"])
            elif it["kind"] == "bare":
                it["names"] = SS.select(recs, it["hide"])
            out.append(it)
        items[str(int(f))] = out
    fs = sorted(int(f) for f in frames)
    screen = fs if ((doc.get("lit") or {}).get("screen") or doc.get("screen")) else None
    req = {"items": items, "keep": {"materials": keep_materials(doc, keep), "frames": fs} if keep else None,
           "screen": screen}
    if outputs:
        req["looks"] = {"outputs": {o: look_jobs(doc, o, fs, items) for o in outputs}}
    return req


# ================================================================================================= the looks
def look_at(doc, output, frame, shot=None):
    """(name, kind, spec) of the look `frame` of `output` is drawn in, as styles.Looks.spec_at: a [[medium]] window's
    ("medium <i>") when one holds the frame with a look for the output, else the shot's (`shot`, a plate's, else the one
    in the cut); (shot name, None, None) without a look, (None, None, None) outside the cut."""
    table = doc.get("shots") or []
    entry = SS.shot_at(table, frame) if shot is None else next((e for e in table if e["name"] == shot), None)
    w = MD.at(doc.get("medium") or [], frame)
    if w is not None:
        look = (w.get("styles") or {}).get(output)
        if look:
            kind = next(iter(look))
            return f"medium {w['index']}", kind, look[kind]
    if entry is None:
        return None, None, None
    look = (entry.get("styles") or {}).get(output)
    if not look:
        return entry["name"], None, None
    kind = next(iter(look))
    return entry["name"], kind, look[kind]


def look_sort(doc, kind, spec, hidden=()):
    """The objects of a look as styles.Looks sorts them on entering it, from the bake's records (`objects`, in the
    scene's order): a silhouette's {hide, subject, hard, soft, type, knock}, a vector look's {hide, subject, type}.
    `hidden` (a glitch's bare frame) are hidden besides: out of every class but `hide`."""
    objs = doc.get("objects") or []
    recs = [SS.Obj(o["name"], tuple(o.get("collections") or ()), frozenset(o.get("props") or ())) for o in objs]
    if kind == "vector":
        cl = SS.vector_sort(recs, spec, {o["name"] for o in objs if o.get("text")})
    else:
        if any("soft" not in o for o in objs):
            raise EngineError("the bake predates mkr's looks (no soft accents in its object records): mk build --bake")
        hide = set(SS.select(recs, spec["hide"])) - set(SS.select(recs, spec["keep"]))
        knock = set(SS.select(recs, spec["knockout"]["objects"])) if spec["knockout"] else set()
        accent = set(SS.select(recs, spec["accent"]))
        cl = {k: [] for k in ("hide", "subject", "hard", "soft", "type", "knock")}
        for o in objs:
            n = o["name"]
            cl["hide" if n in hide else "knock" if n in knock else "type" if o.get("text") else
               ("soft" if o["soft"] else "hard") if n in accent else "subject"].append(n)
    gone = set(hidden) & {o["name"] for o in objs}
    if gone:
        cl = {k: v + [n for n in hidden if n in gone and n not in v] if k == "hide" else [n for n in v if n not in gone]
              for k, v in cl.items()}
    return cl


def _linear(rgb):
    return [SS.srgb_to_linear(float(c)) for c in rgb]


def look_set(doc, kind, spec, hidden=()):
    """mkr's record of a look's passes (looks.rs): the objects each pass draws and how. A silhouette: subject and hard
    (the flat pass, in their colours, linear as the objects' colours are set), soft (alpha and the accent weight),
    type (emission) and knock (the knock-out objects', with a `knockout` table); a vector look: subject (the ids, depth,
    light and texture passes), type, the material ids and whether the texture pass is drawn."""
    cl = look_sort(doc, kind, spec, hidden)
    if kind == "silhouette":
        cols = spec["colors"]
        return {"kind": kind, "subject": cl["subject"], "hard": cl["hard"], "soft": cl["soft"], "type": cl["type"],
                "knock": cl["knock"], "subject_color": _linear(cols["subject"]), "accent_color": _linear(cols["accent"]),
                "samples": int(spec["samples"])}
    objs = {o["name"]: o for o in doc.get("objects") or []}
    mats = SS.vector_materials(objs[n].get("materials") or [] for n in cl["subject"])
    return {"kind": kind, "subject": cl["subject"], "type": cl["type"], "ids": {m: i + 1 for i, m in enumerate(mats)},
            "tex": any(t["kind"] in SS.TEX_KINDS for t in spec["tones"].values()) and SS.draws_fills(spec),
            "light": [float(v) for v in spec["light"]], "samples": 16}


def look_jobs(doc, output, frames, items):
    """The looks of one output's frames and layers, for mkr to draw the passes of: {"sets": [look_set], "jobs": [{id,
    role (frame, plate, bare or matte), frame, shot, key (a bare frame's), set, camera, look (its name)}], "frames":
    {frame: job id}}. A frame of a silhouette or vector look gets a job; so do a plate or bare frame seen through one and a
    matte (a matte's shot without a flat look gets none: Blender stops on it). The items of `items` seen through a look
    get "looks" {output: job id}, a figure "look_names" {output: names} instead (its coverage less what the look hides).
    A frame or layer in a reflection is Blender's."""
    table = {e["name"]: e for e in doc.get("shots") or []}
    sets, keys, jobs = [], {}, []

    def job(role, frame, shot, name, kind, spec, hidden=(), key=None):
        camera = ((table.get(shot) or {}).get("cameras") or {}).get(output)
        if camera is None:
            return None
        sk = (name, kind, tuple(sorted(set(hidden))))
        if sk not in keys:
            keys[sk] = len(sets)
            sets.append(look_set(doc, kind, spec, hidden))
        jobs.append({"id": len(jobs), "role": role, "frame": int(frame), "shot": shot, "set": keys[sk],
                     "camera": camera, "look": name, **({"key": key} if key is not None else {})})
        return len(jobs) - 1
    at = {}
    for f in frames:
        name, kind, spec = look_at(doc, output, f)
        if kind in LOOK_KINDS:
            j = job("frame", f, SS.shot_at(list(table.values()), f)["name"], name, kind, spec)
            if j is not None:
                at[str(int(f))] = j
    for f, its in items.items():
        for it in its:
            if it["kind"] not in (*LOOK_ROLES, "figure"):
                continue
            name, kind, spec = look_at(doc, output, int(f), shot=it["shot"])
            if kind not in LOOK_KINDS:
                continue
            if it["kind"] == "figure":
                hid = set(look_sort(doc, kind, spec)["hide"])
                it.setdefault("look_names", {})[output] = [n for n in it["names"] if n not in hid]
                continue
            j = job(it["kind"], f, it["shot"], name, kind, spec, hidden=it.get("names") if it["kind"] == "bare" else (),
                    key=it.get("key"))
            if j is not None:
                it.setdefault("looks", {})[output] = j
    return {"sets": sets, "jobs": jobs, "frames": at}


def compose_jobs(doc, blob, output, looks, size, percent, frames_dir, ids):
    """{job id: what mkmmd.render.compose.finish_look needs} for the look jobs `ids` of `output` (`looks`: its
    `look_jobs`): the look and its spec, the files it writes (under `frames_dir`), a silhouette's tint values and
    whether it has hard accents, a vector look's table, the rings live on its frame with their centres through its
    camera (frame fractions of a `size` frame at `percent`) and its clip seconds."""
    w, h = PL.render_size(size, percent)
    tints = {(t["object"], t["prop"]): t["offset"] for t in (doc.get("looks") or {}).get("tints") or []}
    rtab = (doc.get("rings") or {}).get("table") or {}
    points = (doc.get("rings") or {}).get("points") or {}
    cams = {c["name"]: c for c in doc.get("cameras") or []}
    root = Path(frames_dir)
    out = {}
    for jid in sorted(set(ids)):
        j = looks["jobs"][jid]
        st = looks["sets"][j["set"]]
        f, k = j["frame"], j["frame"] - doc["frame0"]
        name, kind, spec = look_at(doc, output, f, shot=None if j["role"] == "frame" else j["shot"])
        if name != j["look"] or kind not in LOOK_KINDS:
            raise EngineError(f"look job {jid}: frame {f} of {output} is in look {name!r}, not {j['look']!r}")
        if j["role"] == "frame":
            files = [f"{f:05d}.png"]
        elif j["role"] == "plate":
            files = [TR.plate_rel(j["shot"], f)]
        elif j["role"] == "bare":
            files = [TR.bare_rel(j["key"], f)]
        else:
            files = [TR.back_rel(j["shot"], f), TR.matte_rel(j["shot"], f)]
        rec = {"id": jid, "look": kind, "role": j["role"], "files": [str(root / p) for p in files],
               "t": k / doc["fps"]}
        if kind == "silhouette":
            missing = [(t["object"], t["prop"]) for t in spec["tint"] if (t["object"], t["prop"]) not in tints]
            if missing:
                raise EngineError(f"look {name}: the bake has no values of the tint holders {missing}: mk build --bake")
            # a screen word reversed out of the figure uses the figure's own ink when the look has no knockout table
            rec.update(spec=dict(spec, knockout=spec["knockout"] or {"objects": [], "color": spec["colors"]["subject"]}),
                       tints=[float(blob[tints[(t["object"], t["prop"])] + k]) for t in spec["tint"]],
                       hard=bool(st["hard"]))
        else:
            mats = sorted(st["ids"], key=st["ids"].get)
            tones, groups = SS.vector_table(mats, spec)
            rings, centres, rt = [], {}, rec["t"]
            if rtab:
                rt = (f - rtab["frame0"]) / rtab["fps"]
                rings = rtab.get("rings") or []
                cam = cams[j["camera"]]
                for r in RG.live(rings, rt):
                    if r["center"]["mode"] == "frame":
                        centres[r["index"]] = tuple(r["center"]["at"])
                    else:
                        o = points[str(r["index"])] + 3 * k
                        p = np.asarray(blob[o:o + 3], np.float64)
                        centres[r["index"]] = None if not np.isfinite(p).all() else PL.project(blob, cam, k, (w, h), p)
            rec.update(spec=spec, table=[list(tones), list(groups)], rings=rings, centres=centres, ring_t=rt)
        out[jid] = rec
    return out


def _done(path):
    """Whether a layer file is finished (on disk and not an empty claim)."""
    return path.is_file() and path.stat().st_size > 0


def split(req, out_plan, frames_dir, output=None):
    """(mine, theirs) for one output: `mine` the request mkr draws (`draw`), every layer not on disk that `plan` gave it
    and, with `output`, the look jobs of its frames and layers; `theirs` what Blender's job draws: "demands" {frame:
    [items]}, "keep" and "screen" (mkr's frames whose keep mask or screen type mkr cannot draw), with "mk_keep" /
    "mk_screen" the frames whose own mkr draws. A frame's screen type is on disk with its screen layer and, where
    `plan` says it has negative type ("negative"), its negative layer."""
    root = Path(frames_dir)
    verdicts = out_plan.get("layers") or {}
    mine = {"items": {}, "keep": None, "screen": None}
    theirs = {"demands": {}}
    for f, items in req["items"].items():
        why = verdicts.get(int(f)) or ["mkr info did not sort it"] * len(items)
        for item, reason in zip(items, why):
            if _done(root / TR.rel_paths(item, f)[-1]):
                continue
            if reason is None:
                mine["items"].setdefault(f, []).append(item)
            else:
                theirs["demands"].setdefault(int(f), []).append({k: v for k, v in item.items()
                                                                  if k not in ("names", "looks", "look_names")})
    negative = set((out_plan.get("screen") or {}).get("negative") or ())

    def done(kind, f):
        if kind == "keep":
            return _done(root / KP.rel(f))
        return _done(root / SRT.layer_rel(f)) and (f not in negative or _done(root / SRT.negative_rel(f)))

    for kind in ("keep", "screen"):
        sorted_ = out_plan.get(kind) or {}
        ours = sorted(sorted_.get("draw", []))
        todo = [f for f in ours if not done(kind, f)]
        theirs["mk_" + kind] = ours
        theirs[kind] = sorted(f for f in (sorted_.get("blender") or {}) if not done(kind, f))
        if kind == "keep" and req["keep"] and todo:
            mine["keep"] = {"materials": req["keep"]["materials"], "frames": todo}
        elif kind == "screen" and todo:
            mine["screen"] = todo
    looks = ((req.get("looks") or {}).get("outputs") or {}).get(output) if output else None
    if looks:
        draw = {int(f) for f in out_plan.get("draw") or ()}
        at = {f: j for f, j in looks["frames"].items() if int(f) in draw and not _done(root / f"{int(f):05d}.png")}
        mine["looks"] = {"outputs": {output: {"sets": looks["sets"], "jobs": looks["jobs"], "frames": at}}}
    return mine, theirs


def look_ids(req, output):
    """The look jobs a request (`split`'s `mine`) has mkr draw for `output`: its frames' and its layers'."""
    looks = ((req.get("looks") or {}).get("outputs") or {}).get(output)
    if not looks:
        return []
    ids = set(looks["frames"].values())
    for items in req.get("items", {}).values():
        for it in items:
            if output in (it.get("looks") or {}):
                ids.add(it["looks"][output])
    return sorted(ids)


def empty(req):
    """Whether a request asks for nothing."""
    return not req or not (req.get("items") or req.get("keep") or req.get("screen") or
                           any(lk["frames"] for lk in ((req.get("looks") or {}).get("outputs") or {}).values()))


# ================================================================================================= the runs
def plan(bake, outputs, frames, layers=None):
    """{output: {"draw": [frames mkr draws], "blender": {frame: why}}} for these outputs and frames, plus what mkr
    cannot draw at all (`unsupported`). With `layers` (a `request`) each output also has "layers" {frame: [None (mkr
    draws it) or why Blender must, per item]}, "keep" and "screen" {"draw": [mkr's frames whose layer mkr draws],
    "blender": {frame: why}} (screen: only the frames with screen type, and "negative": those of "draw" with negative
    type on screen, whose negative layer mkr draws too)."""
    argv = ["info", "--bake", str(bake), "--frames", spec(frames)]
    for name in outputs:
        argv += ["--output", name]
    doc = _call(argv, 300, layers)
    outs = doc.get("outputs") or {}
    missing = [n for n in outputs if n not in outs]
    if missing:
        raise EngineError(f"mkr info said nothing about output(s) {', '.join(missing)}")
    res = {}
    for n in outputs:
        o = outs[n]
        res[n] = {"draw": sorted(int(f) for f in o.get("draw", [])),
                  "blender": {int(k): v for k, v in (o.get("blender") or {}).items()}}
        if layers is not None:
            res[n]["layers"] = {int(k): v for k, v in (o.get("layers") or {}).items()}
            for kind in ("keep", "screen"):
                s = o.get(kind) or {}
                res[n][kind] = {"draw": sorted(int(f) for f in s.get("draw", [])),
                                "blender": {int(k): v for k, v in (s.get("blender") or {}).items()}}
            res[n]["screen"]["negative"] = sorted(int(f) for f in (o.get("screen") or {}).get("negative") or ())
    return {"outputs": res, "unsupported": doc.get("unsupported") or {}}


def draw(bake, output, frames, out_dir, cfg, layers=None, timeout=3600 * 12, extra=(), looks=None, sampling=None):
    """Render `frames` of `output` into out_dir/<frame>.png and the layers of `layers` (a request: `split`'s `mine`)
    beside them, with the preset in `cfg` (samples, percent, motion_blur, shutter) and any `extra` mkr arguments
    (`--params FILE`), or with `sampling` ({samples, params}: FINAL) mkr's own samples and tunables for the preset's
    (the tunables of an `extra` --params over them); mkr's report. `looks` ({job id: `compose_jobs` record}) are the look
    jobs of `layers`: mkr draws their passes into a Composer's spool, whose workers finish them meanwhile (the report's
    "looks")."""
    if not frames and empty(layers):
        return {"rendered": 0}
    samples = sampling["samples"] if sampling is not None else cfg["samples"]
    params = dict(sampling["params"]) if sampling is not None else None
    extra = [str(a) for a in extra]
    if params is not None:
        # an extra --params (FILE or =FILE) goes over the sampling's tunables: mkr takes a single --params
        extra = [p for a in extra for p in (["--params", a.split("=", 1)[1]] if a.startswith("--params=") else [a])]
        if "--params" in extra:
            i = extra.index("--params")
            try:
                params.update(json.loads(Path(extra[i + 1]).read_text(encoding="utf-8")))
            except (IndexError, OSError, TypeError, ValueError) as e:
                raise EngineError(f"mkr --params: {e}") from None
            extra = extra[:i] + extra[i + 2:]
    argv = ["render", "--bake", str(bake), "--output", output, "--out", str(out_dir),
            "--samples", str(int(samples)), "--percent", str(int(cfg["percent"]))]
    if frames:
        argv += ["--frames", spec(frames)]
    if cfg.get("motion_blur"):
        argv += ["--motion-blur", "--shutter", str(float(cfg.get("shutter", 0.35)))]
    argv += extra
    if not looks:
        return _call(argv, timeout, None if empty(layers) else layers, params=params)
    composer = Composer(looks)
    try:
        layers = dict(layers, looks=dict(layers["looks"], spool=composer.spool, limit=composer.limit))
        rep = _call(argv, timeout, layers, composer, params=params)
    finally:
        stats = composer.close()
    if composer.errors:
        raise EngineError(f"composing mkr's look passes: {composer.errors[0]}")
    rep["looks"] = stats
    return rep


def compose_workers():
    """How many processes compose look passes: COMPOSE_JOBS at most, no more than the CPU threads but two (mkr's own
    threads), and no more than MEMORY_SHARE of the memory available holds at COMPOSE_PEAK each; unknown memory: one."""
    avail = MEM.available()
    if avail is None:
        return 1
    return max(1, min(COMPOSE_JOBS, (os.cpu_count() or 2) - 2, int(avail * MEMORY_SHARE // COMPOSE_PEAK)))


class Composer:
    """Finishes mkr's look passes while it draws the next (docs/design.md: Rendering, mkr's looks): mkr writes each
    look job's passes to <spool>/<job>.look (whole, renamed into place) and waits while `limit` of them are there;
    worker processes compose each (mkmmd.render.compose.finish_look: the frame or layer files, then the pass file is
    deleted). The spool is in memory (/dev/shm) where the system has it."""

    def __init__(self, jobs, workers=None):
        import multiprocessing
        self.jobs = {int(k): v for k, v in jobs.items()}
        self.workers = workers or compose_workers()
        self.limit = 2 * self.workers
        MEM.sweep("mkr_looks", [SHM, tempfile.gettempdir()])        # spools of runs killed outright
        shm = SHM if os.path.isdir(SHM) and os.access(SHM, os.W_OK) else None
        self.spool = tempfile.mkdtemp(prefix=MEM.scratch_prefix("mkr_looks"), dir=shm)
        self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))
        self.seen, self.futs, self.errors = set(), [], []
        self.lock = threading.Lock()
        self.t0, self.done, self.cpu = time.time(), 0, 0.0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _scan(self):
        for name in sorted(os.listdir(self.spool)):
            if not name.endswith(".look") or name in self.seen:
                continue
            self.seen.add(name)
            job = self.jobs.get(int(name[:-5]))
            if job is None:
                with self.lock:
                    self.errors.append(f"mkr wrote passes of job {name[:-5]}, which nobody asked for")
                continue
            fut = self.pool.submit(_finish_timed, os.path.join(self.spool, name), job)
            fut.add_done_callback(self._finished)
            self.futs.append(fut)

    def _finished(self, fut):
        with self.lock:
            try:
                self.cpu += fut.result()
                self.done += 1
            except Exception as e:  # noqa: BLE001 - a worker's failure is reported with the run
                self.errors.append(f"{type(e).__name__}: {e}")

    def _watch(self):
        while not self.stop.is_set():
            self._scan()
            time.sleep(0.02)

    def failed(self):
        with self.lock:
            return bool(self.errors)

    def close(self):
        """Finish what mkr left (it has exited) and clean up: {composed, seconds, cpu_seconds, workers}."""
        self.stop.set()
        self.thread.join()
        try:
            if not self.errors:
                self._scan()
            for fut in self.futs:
                try:
                    fut.result()
                except Exception:  # noqa: BLE001 - recorded by _finished
                    pass
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
            shutil.rmtree(self.spool, ignore_errors=True)
        missing = sorted(set(self.jobs) - {int(n[:-5]) for n in self.seen})
        if missing and not self.errors:
            self.errors.append(f"mkr wrote no passes for look job(s) {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        return {"composed": self.done, "seconds": round(time.time() - self.t0, 1), "cpu_seconds": round(self.cpu, 1),
                "workers": self.workers}


def _finish_timed(path, job):
    """A worker's compose of one pass file (mkmmd.render.compose.finish_look); its CPU seconds."""
    t0 = time.process_time()
    CO.finish_look(path, job)
    return time.process_time() - t0
