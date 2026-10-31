"""Parity of mkr's frames with Blender's (docs/design.md: Parity): metrics, contact sheets, the frames of a project
rendered by mkr against Blender's, and hill climbing over mkr's tunables.

Metrics compare a test frame with its reference over the pixels a mask keeps (every pixel without one), in 8-bit
levels (a 16-bit PNG counts as levels with fractions, value * 255 / 65535):
  mean_abs   mean |test - ref| over pixels and channels
  bias       [r, g, b]: mean test - ref per channel (positive: the test is brighter)
  over8      % of pixels whose largest channel error is more than 8 levels; over32 the same over 32
  p99, max   the 99th percentile and the largest per-pixel error (the largest channel's), 1/8-level steps
  rmse, psnr root mean square error over pixels and channels; 10 log10(255^2 / mse) dB (None: identical)
Over several frames they pool every pixel of every frame (each frame weighs as many pixels as it has); the "1:1" bar is
ACCEPT on the pooled numbers.

`compare` matches the frames of two folders by file name (`00091.png` and `91.png` are frame 91), runs the frames in
parallel threads (Pillow decodes, OpenCV the 16-bit files; numpy measures, in integers for two 8-bit frames) and draws
contact sheets: per frame the reference, the test and the error amplified as heat (`gain` x the per-pixel error through
inferno: 32 levels saturate at the default 8), with an optional fourth panel of the signed luma error (red: the test is
brighter, blue: darker).

`plan` and `render_compare` are `mk parity frames`: the lit frames of a project's outputs drawn by mkr and measured;
`tune` climbs mkr's tunables (`Objective`, `coordinate`, `nelder_mead`) on any `evaluate(params_file, draw)`."""
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..core import jsonx
from ..core import medium as MD
from ..core import shotstyle as SS
from . import mkr as MKR

BINS = 8                                  # histogram bins per level: p99 and max in 1/8-level steps (exact for 8-bit)
NBIN = 256 * BINS + 1
ACCEPT = {"mean_abs": 1.0, "over8": 0.5}  # the "1:1" bar: mean |error| <= 1 level, <= 0.5 % of pixels off by > 8
METRICS = ("mean_abs", "rmse", "over8", "over32", "p99", "max", "bias_abs", "psnr")   # what tune can minimise
POOLED = ("mean_abs", "over8", "over32", "bias_abs")                                   # ...and pool over pixels
LOG_DECIMALS = 6                          # tune.jsonl keeps params and scores to this many decimals
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
PNG = b"\x89PNG\r\n\x1a\n"


class ParityError(ValueError):
    """Frames that cannot be compared."""


def read(path):
    """An image as (h, w, 3) RGB levels: uint8 for 8-bit files, float32 levels 0..255 for 16-bit PNGs (alpha dropped,
    grey spread to three channels). Pillow reads 8-bit files (it keeps quiet about the duplicate eXIf chunks Blender
    writes), OpenCV the 16-bit ones (Pillow would cut them to 8 bits)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(26)
        if head[:8] == PNG and len(head) == 26 and head[24] == 16:
            im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if im is None:
                raise ParityError(f"cannot read {path}")
            im = np.repeat(im[:, :, None], 3, axis=2) if im.ndim == 2 else im[:, :, 2::-1]
            return im.astype(np.float32) * np.float32(255.0 / 65535.0)
        with Image.open(path) as img:
            return np.asarray(img if img.mode == "RGB" else img.convert("RGB"))
    except (OSError, ValueError) as e:
        raise ParityError(f"cannot read {path}: {e}") from None


def frame_key(stem):
    return int(stem) if stem.isdigit() else stem


def frame_files(d):
    """{frame: path} of the PNG frames of a folder, by file name (an integer name is its frame number); empty claim
    files of a render in progress are left out."""
    d = Path(d)
    if not d.is_dir():
        raise ParityError(f"no folder {d}")
    return {frame_key(p.stem): p for p in sorted(d.glob("*.png")) if p.stat().st_size > 0}


def region_mask(shape, region):
    """A box (x0, y0, x1, y1) as a mask: fractions of the width and height when every number is <= 1, else pixels."""
    h, w = shape
    x0, y0, x1, y1 = (float(v) for v in region)
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
    m = np.zeros((h, w), bool)
    m[max(0, round(y0)):max(0, round(y1)), max(0, round(x0)):max(0, round(x1))] = True
    return m


def pixels(key, ref, test, mask=None, region=None, figure=None):
    """The pixels a frame is measured on (mask_for), and with `figure` (levels) only those where either frame has a
    channel brighter than that: the figure, the black void around it left out."""
    m = mask_for(key, ref.shape[:2], mask, region)
    if figure is None:
        return m
    fig = (ref.max(axis=2) > figure) | (test.max(axis=2) > figure)
    return fig if m is None else m & fig


def mask_for(key, shape, mask=None, region=None):
    """The pixels a frame is measured on: a mask image (white counts; a folder holds one per frame, by name), a region
    box, both (their intersection) or None (every pixel)."""
    m = None
    if mask is not None:
        p = Path(mask)
        if p.is_dir():
            files = frame_files(p)
            if key not in files:
                raise ParityError(f"no mask for frame {key} in {p}")
            p = files[key]
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ParityError(f"cannot read the mask {p}")
        if im.shape != tuple(shape):
            raise ParityError(f"the mask {p} is {im.shape[1]}x{im.shape[0]}, the frame {shape[1]}x{shape[0]}")
        m = im > 127
    if region is not None:
        r = region_mask(shape, region)
        m = r if m is None else m & r
    return m


# ================================================================================================= metrics
class Stats:
    """Sums that pool exactly over frames: pixel count, per-channel sums of |d| and d, the sum of d^2, counts of pixels
    over 8 and 32 levels and the histogram of the per-pixel error (d = test - ref, levels)."""
    __slots__ = ("abs", "hist", "n", "over", "signed", "sq")

    def __init__(self):
        self.n, self.sq = 0, 0.0
        self.abs, self.signed = np.zeros(3), np.zeros(3)
        self.over = np.zeros(2, np.int64)
        self.hist = np.zeros(NBIN, np.int64)

    def add(self, other):
        self.n += other.n
        self.sq += other.sq
        self.abs += other.abs
        self.signed += other.signed
        self.over += other.over
        self.hist += other.hist
        return self


def stats(ref, test, mask=None):
    """Stats of `test` against `ref` ((h, w, 3) levels) over the pixels of `mask` (bool (h, w); None: all). Two 8-bit
    frames are measured in integers (exact); anything else in float32."""
    if ref.shape != test.shape:
        raise ParityError(f"sizes differ: reference {ref.shape[1]}x{ref.shape[0]}, test {test.shape[1]}x{test.shape[0]}")
    whole = ref.dtype == np.uint8 and test.dtype == np.uint8
    d = test.astype(np.int16 if whole else np.float32)
    d -= ref
    d = d[mask] if mask is not None else d.reshape(-1, 3)
    a = np.abs(d)
    e = a.max(axis=1)
    acc = np.int64 if whole else np.float64
    st = Stats()
    st.n = int(e.size)
    st.abs = a.sum(axis=0, dtype=acc).astype(np.float64)
    st.signed = d.sum(axis=0, dtype=acc).astype(np.float64)
    st.sq = float(np.square(a, dtype=np.int32 if whole else np.float32).sum(dtype=acc))
    st.over = np.array([np.count_nonzero(e > 8), np.count_nonzero(e > 32)], np.int64)
    bins = e.astype(np.intp) * BINS if whole else np.minimum(np.rint(e * BINS), NBIN - 1).astype(np.intp)
    st.hist = np.bincount(bins, minlength=NBIN)
    return st


def metrics(st):
    """The metrics (module doc) of a Stats."""
    n = st.n
    if n == 0:
        return {"pixels": 0}
    mse = st.sq / (3 * n)
    nz = np.flatnonzero(st.hist)
    return {"pixels": n, "mean_abs": float(st.abs.sum() / (3 * n)), "bias": [float(b) for b in st.signed / n],
            "over8": 100.0 * float(st.over[0]) / n, "over32": 100.0 * float(st.over[1]) / n,
            "p99": float(np.searchsorted(np.cumsum(st.hist), 0.99 * n)) / BINS,
            "max": float(nz[-1]) / BINS if nz.size else 0.0,
            "rmse": math.sqrt(mse), "psnr": 10.0 * math.log10(255.0 ** 2 / mse) if mse > 0 else None}


def passes(m):
    """Whether pooled metrics meet the "1:1" bar (ACCEPT)."""
    return bool(m.get("pixels")) and m["mean_abs"] <= ACCEPT["mean_abs"] and m["over8"] <= ACCEPT["over8"]


def score(m, metric):
    """The number tune minimises for one frame's metrics: the metric itself, mean |bias| for bias_abs, -psnr for psnr
    (an identical frame counts as 100 dB)."""
    if metric == "bias_abs":
        return float(np.mean(np.abs(m["bias"])))
    if metric == "psnr":
        return -(m["psnr"] if m["psnr"] is not None else 100.0)
    return float(m[metric])


# ================================================================================================= pictures
def heat(ref, test, mask=None, gain=8.0):
    """The per-pixel error (the largest channel's) times `gain` through inferno, BGR; pixels outside the mask dimmed."""
    e = np.abs(test.astype(np.float32) - ref.astype(np.float32)).max(axis=2)
    out = cv2.applyColorMap(np.clip(e * gain, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if mask is not None:
        out[~mask] //= 4
    return out


def signed_heat(ref, test, mask=None, gain=8.0):
    """The signed luma error times `gain`, BGR: red where the test is brighter, blue where it is darker."""
    dl = (test.astype(np.float32) - ref.astype(np.float32)) @ LUMA
    out = np.zeros(dl.shape + (3,), np.uint8)
    out[..., 2] = np.clip(dl * gain, 0, 255).astype(np.uint8)
    out[..., 0] = np.clip(-dl * gain, 0, 255).astype(np.uint8)
    if mask is not None:
        out[~mask] //= 4
    return out


def _thumb(bgr, box):
    h, w = bgr.shape[:2]
    s = min(box / w, box / h, 1.0)
    if s >= 1.0:
        return bgr
    return cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _to_bgr8(im):
    return np.ascontiguousarray(np.clip(np.rint(im), 0, 255).astype(np.uint8)[:, :, ::-1])


def _label(m):
    if not m.get("pixels"):
        return "no pixels"
    psnr = f"{m['psnr']:.1f} dB" if m["psnr"] is not None else "identical"
    b = m["bias"]
    return (f"mean {m['mean_abs']:.2f}  >8 {m['over8']:.2f}%  >32 {m['over32']:.2f}%  p99 {m['p99']:g}  "
            f"bias {b[0]:+.2f} {b[1]:+.2f} {b[2]:+.2f}  {psnr}")


def _row(key, ref, test, mask, m, box, gain, signed):
    panels = [_to_bgr8(ref), _to_bgr8(test), heat(ref, test, mask, gain)]
    if signed:
        panels.append(signed_heat(ref, test, mask, gain))
    panels = [_thumb(p, box) for p in panels]
    gap = 6
    w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    h = max(p.shape[0] for p in panels)
    bar = 22
    row = np.full((h + bar, w, 3), 24, np.uint8)
    x = 0
    for p in panels:
        row[bar:bar + p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    name = f"{key:05d}" if isinstance(key, int) else str(key)
    cv2.putText(row, f"{name}  {_label(m)}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return row


def sheets(rows, out, per_sheet=6, name="sheet"):
    """Stack rows (BGR) into PNG sheets of `per_sheet` rows each, as wide as the widest; their paths."""
    paths = []
    for i in range(0, len(rows), per_sheet):
        part = rows[i:i + per_sheet]
        w = max(r.shape[1] for r in part)
        img = np.full((sum(r.shape[0] for r in part) + 4 * (len(part) - 1), w, 3), 12, np.uint8)
        y = 0
        for r in part:
            img[y:y + r.shape[0], :r.shape[1]] = r
            y += r.shape[0] + 4
        p = Path(out) / f"{name}_{i // per_sheet + 1:02d}.png"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        paths.append(str(p))
    return paths


# ================================================================================================= compare
def _sheet_frames(per_frame, limit):
    """At most `limit` frames for the sheets: all when they fit, else the worst half by mean |error| and the rest
    spread evenly over the others, in frame order."""
    keys = list(per_frame)
    if len(keys) <= limit:
        return keys
    worst = sorted(keys, key=lambda k: -per_frame[k].get("mean_abs", 0.0))[:limit // 2]
    rest = [k for k in keys if k not in set(worst)]
    step = len(rest) / (limit - len(worst))
    spread = [rest[int(i * step)] for i in range(limit - len(worst))]
    chosen = set(worst) | set(spread)
    return [k for k in keys if k in chosen]


def _listed(keys):
    """Frames as a frame spec ("91:94,96": integer names) or a sorted list of names."""
    keys = list(keys)
    if all(isinstance(k, int) for k in keys):
        return MKR.spec(keys)
    return sorted(str(k) for k in keys)


def compare(ref_dir, test_dir, out=None, frames=None, mask=None, region=None, gain=8.0, signed=False, box=480,
            per_sheet=6, sheet_limit=60, threads=None, draw=True, figure=None):
    """Compare the frames of `test_dir` with those of `ref_dir` (module doc). `frames` limits them (frame numbers);
    `out` gets summary.json and the sheets (none when `draw` is false or `out` is None). Returns the summary: overall
    (pooled) metrics and whether they pass, per-frame metrics, the worst frames, the frames only one folder has and
    the frames that could not be compared (why)."""
    t0 = time.time()
    refs, tests = frame_files(ref_dir), frame_files(test_dir)
    want = set(frames) if frames is not None else None
    keys = sorted((k for k in refs.keys() & tests.keys() if want is None or k in want),
                  key=lambda k: (isinstance(k, str), k))
    only_ref = _listed(k for k in refs.keys() - tests.keys() if want is None or k in want)
    only_test = _listed(k for k in tests.keys() - refs.keys() if want is None or k in want)
    absent = _listed(k for k in want if k not in refs and k not in tests) if want is not None else ""

    def one(k):
        try:
            ref, test = read(refs[k]), read(tests[k])
            m = pixels(k, ref, test, mask, region, figure)
            return k, stats(ref, test, m), None
        except ParityError as e:
            return k, None, str(e)

    workers = threads or min(6, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        done = list(ex.map(one, keys))
    total, per_frame, errors = Stats(), {}, {}
    for k, st, err in done:
        if err is not None:
            errors[k] = err
            continue
        total.add(st)
        per_frame[k] = metrics(st)
    overall = metrics(total)
    worst = sorted(per_frame, key=lambda k: -per_frame[k].get("mean_abs", 0.0))[:5]
    summary = {"ref": str(ref_dir), "test": str(test_dir), "frames": len(per_frame), "overall": overall,
               "pass": passes(overall), "accept": dict(ACCEPT),
               "worst": [{"frame": k, "mean_abs": per_frame[k]["mean_abs"], "over8": per_frame[k]["over8"]} for k in worst],
               "only_ref": only_ref, "only_test": only_test, "absent": absent, "errors": errors,
               "mask": str(mask) if mask is not None else None, "region": list(region) if region is not None else None,
               "figure": figure, "per_frame": per_frame, "_stats": total}
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("sheet_*.png"):
            old.unlink()
        if draw and per_frame:
            chosen = _sheet_frames(per_frame, sheet_limit)

            def row(k):
                ref, test = read(refs[k]), read(tests[k])
                return _row(k, ref, test, pixels(k, ref, test, mask, region, figure), per_frame[k], box, gain, signed)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                rows = list(ex.map(row, chosen))
            summary["sheets"] = sheets(rows, out, per_sheet)
            summary["sheet_frames"] = chosen
        summary["seconds"] = round(time.time() - t0, 2)
        summary["summary"] = str(out / "summary.json")
        (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1), encoding="utf-8")
    summary.setdefault("seconds", round(time.time() - t0, 2))
    return summary


def _jsonable(summary):
    return jsonx.to_jsonable({k: v for k, v in summary.items() if not k.startswith("_")}, 6)


def brief(summary):
    """A summary without its per-frame table and pooled sums (what the commands print)."""
    return {k: v for k, v in summary.items() if k != "per_frame" and not k.startswith("_")}


def pool(summaries):
    """Pooled metrics of several summaries (every pixel of every frame of each)."""
    total = Stats()
    for s in summaries:
        total.add(s["_stats"])
    return metrics(total)


# ================================================================================================= a project's frames
def looked(doc, output, frames):
    """{frame: why} of `frames` whose picture for `output` is not the lit scene Blender draws with EEVEE: a shot with a
    render-time look (the bake's shot table: `styles`), a [[medium]] window with a look of its own, a fisheye camera
    (Cycles draws it). Parity leaves them out."""
    shots = [e for e in doc.get("shots") or [] if not e.get("plate")]
    cams = {c["name"]: c for c in doc.get("cameras") or []}
    media = doc.get("medium") or []
    out = {}
    for f in frames:
        win = MD.at(media, f)
        if win is not None and (win.get("styles") or {}).get(output):
            out[f] = f"medium {win['index']}: a {next(iter(win['styles'][output]))} look"
            continue
        e = SS.shot_at(shots, f)
        if e is None:
            continue
        look = (e.get("styles") or {}).get(output)
        if look:
            out[f] = f"shot {e['name']}: a {next(iter(look))} look"
        elif (cams.get((e.get("cameras") or {}).get(output)) or {}).get("fisheye"):
            out[f] = f"shot {e['name']}: a fisheye camera (Cycles)"
    return out


def percent_of(path, size):
    """The resolution percentage a reference frame was rendered at, for an output of `size`."""
    with Image.open(path) as im:
        w, h = im.size
    p = round(100 * w / size[0])
    if p < 1 or (size[0] * p // 100, size[1] * p // 100) != (w, h):
        raise ParityError(f"{path} is {w}x{h}: not the {size[0]}x{size[1]} output at a whole percentage")
    return p


def plan(proj, outputs, frames=None, refs=None, preset="final"):
    """What `mk parity frames` draws and compares: (bake folder, {output: {ref (Blender's frames folder), frames (the
    lit frames it has), skipped ({why: frame spec}), missing (lit frames asked for that it lacks), outside (asked for,
    outside the clip), percent (the references')}}). `frames` None: every frame the reference folder has. `refs` a
    folder of frames, or of <output>/ folders; default <project>/renders/<preset>/<output>."""
    bake = proj.blend.with_suffix(".bake") if proj.blend else None
    try:
        doc = json.loads((bake / "bake.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as e:
        raise ParityError(f"no bake of the project's scene ({e}): run `mk build` (or `mk build --bake`)") from None
    first, n = int(doc["frame0"]), int(doc["frames"])
    out = {}
    for o in outputs:
        size = next((x["size"] for x in doc.get("outputs") or [] if x["name"] == o), None)
        if size is None:
            have = ", ".join(x["name"] for x in doc.get("outputs") or [])
            raise ParityError(f"the bake has no output {o!r} (it has {have})")
        if refs is None:
            ref = proj.root / "renders" / preset / o
        else:
            r = Path(refs).expanduser().resolve()
            ref = r / o if (r / o).is_dir() else r
            if ref == r and len(outputs) > 1:
                raise ParityError(f"--refs {r} has no {o}/ folder (several outputs need one folder each)")
        have = {k: p for k, p in frame_files(ref).items() if isinstance(k, int)}
        want = sorted(have) if frames is None else sorted({int(f) for f in frames})
        clip = [f for f in want if first <= f < first + n]
        skip = looked(doc, o, clip)
        lit = [f for f in clip if f not in skip and f in have]
        groups = {}
        for f, why in skip.items():
            groups.setdefault(why, []).append(f)
        out[o] = {"ref": ref, "frames": lit, "skipped": {why: MKR.spec(fs) for why, fs in groups.items()},
                  "missing": [f for f in clip if f not in skip and f not in have],
                  "outside": [f for f in want if not first <= f < first + n],
                  "percent": percent_of(have[lit[0]], size) if lit else 100}
    return bake, out


def render_compare(bake, plans, run_dir, cfg, extra=(), draw=True, **kw):
    """mkr's frames of each output's plan (`plan`) into run_dir/<output>/frames/, compared with the references (sheets
    and summary.json in run_dir/<output>/compare/ when `draw`): {output: {mkr (its report), summary}}."""
    res = {}
    for o, p in plans.items():
        fd = Path(run_dir) / o / "frames"
        fd.mkdir(parents=True, exist_ok=True)
        for old in fd.glob("*.png"):
            old.unlink()
        rep = MKR.draw(bake, o, p["frames"], fd, dict(cfg, percent=p["percent"]), extra=extra, sampling=cfg.get("sampling"))
        res[o] = {"mkr": rep, "summary": compare(p["ref"], fd, Path(run_dir) / o / "compare" if draw else None,
                                                  frames=p["frames"], draw=draw, **kw)}
    return res


def project_evaluator(bake, plans, scratch, cfg, extra=(), **kw):
    """evaluate(params_file, draw) for tune on a project's frames (`kw`: compare's mask, region, figure):
    [(output, frame, metrics)]."""
    def evaluate(params_file, draw=False):
        res = render_compare(bake, plans, scratch, cfg, extra=["--params", str(params_file), *extra], draw=draw, **kw)
        out = []
        for o, r in res.items():
            s = r["summary"]
            if s["errors"] or s["only_ref"]:
                raise ParityError(f"{o}: mkr left frames out or they cannot be read ({s['only_ref'] or s['errors']})")
            out += [(o, f, m) for f, m in s["per_frame"].items()]
        return out
    return evaluate


def combined(evaluators):
    """One evaluate over several ({group: evaluate}): their frames, each part named `group:part` (Objective sums the
    groups' means)."""
    def evaluate(params_file, draw=False):
        out = []
        for g, ev in evaluators.items():
            out += [(f"{g}:{part}", f, m) for part, f, m in ev(params_file, draw=draw)]
        return out
    return evaluate


# ================================================================================================= hill climbing
def load_space(path):
    """A tune file: {"params": {name: {"start", "step", "min", "max", "int"} | {"choices": [...], "start"}}, "fixed":
    {name: value} (passed as they are), "metric", "method" ("coord" | "nelder-mead"), "iterations", "min_step", "pool"
    (score each part's frames pooled, their pixels weighing alike, as `compare` reports them; then the mean over the
    parts), "weights" ({group: weight}: with a project and lab cases, how much each group's mean counts, 1 unless
    named)}. `start` defaults to mkr's own default (`mkr params`)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ParityError(f"cannot read the tune file {path}: {e}") from None
    params = doc.get("params") or {}
    if not params:
        raise ParityError(f"{path}: no params to tune")
    space = {}
    for name, p in params.items():
        if "choices" in p:
            if not p["choices"]:
                raise ParityError(f"{path}: {name} has no choices")
            space[name] = {"choices": list(p["choices"]), "start": p.get("start")}
        elif "step" in p and float(p["step"]) > 0:
            space[name] = {"start": p.get("start"), "step": float(p["step"]), "min": p.get("min"), "max": p.get("max"),
                           "int": bool(p.get("int", False))}
        else:
            raise ParityError(f"{path}: {name} needs a positive step (or choices)")
    metric = doc.get("metric", "mean_abs")
    if metric not in METRICS:
        raise ParityError(f"{path}: metric {metric!r} is not one of {', '.join(METRICS)}")
    pool = bool(doc.get("pool", False))
    if pool and metric not in POOLED:
        raise ParityError(f"{path}: pool scores {', '.join(POOLED)} only, not {metric!r}")
    weights = doc.get("weights") or {}
    if not isinstance(weights, dict) or not all(isinstance(v, (int, float)) and v > 0 for v in weights.values()):
        raise ParityError(f"{path}: weights must map group names (project, lab) to positive numbers")
    return {"params": space, "fixed": dict(doc.get("fixed") or {}), "metric": metric, "method": doc.get("method", "coord"),
            "iterations": int(doc.get("iterations", 60)), "min_step": doc.get("min_step"), "pool": pool,
            "weights": {k: float(v) for k, v in weights.items()}}


def mkr_params():
    """mkr's tunables and their defaults (`mkr params`: one JSON object)."""
    import subprocess
    r = subprocess.run([str(MKR.binary()), "params"], capture_output=True, text=True, timeout=120, check=False)
    if r.returncode != 0:
        raise ParityError(f"`mkr params` failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
    try:
        return json.loads(r.stdout)
    except ValueError:
        raise ParityError(f"`mkr params` printed no JSON: {r.stdout[-200:]!r}") from None


class Objective:
    """The function the climbers minimise: params -> the mean over every frame of the metric (`score`), through
    `evaluate(params_file, draw)` -> [(part, frame, metrics)]; with `pool`, the mean over the parts of each part's
    frames pooled (weighted by their pixels: a frame's metric as `compare` pools it). Parts named `group:part`
    (`combined`) score as the sum over the groups of each group's mean (times its `weights` entry, 1 unless named), so
    that a project's frames and the lab's weigh as chosen whatever their counts. Points are cached (a revisit costs
    nothing and counts no step); every evaluation is a line of out/tune.jsonl; the best so far is out/best.json and
    out/best_params.json (mkr's --params file). A render that fails scores infinity. `resume`: an earlier run's log
    lines (`read_log`), cached and logged again first, its best the one to beat."""

    def __init__(self, evaluate, metric, space, fixed, out, say=None, pool=False, resume=(), weights=None):
        self.evaluate, self.metric, self.space, self.fixed, self.pool = evaluate, metric, space, fixed, pool
        self.weights = dict(weights or {})
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.say = say or (lambda line: None)
        self.cache, self.n, self.best, self.first, self.done = {}, 0, None, None, 0
        for rec in resume:
            if (rec.get("metric", metric) != metric or bool(rec.get("pool", pool)) != bool(pool)
                    or rec.get("weights", {}) != self.weights):
                raise ParityError(f"the log to resume scored {rec.get('metric')} (pool {rec.get('pool')}, weights "
                                  f"{rec.get('weights', {})}), not {metric} (pool {pool}, weights {self.weights})")
            if set(rec["params"]) != set(space):
                raise ParityError(f"the log to resume tuned {', '.join(sorted(rec['params']))}, not "
                                  f"{', '.join(sorted(space))}")
            value = math.inf if rec.get("score") is None else float(rec["score"])
            self.cache[self.key(rec["params"])] = value
            self.done = max(self.done, int(rec.get("step") or 0))
            self.first = self.first or rec
            if value < (self.best["score"] if self.best else math.inf):
                self.best = dict(rec, score=value)
        self.log = self.out / "tune.jsonl"
        self.log.write_text("".join(json.dumps(rec) + "\n" for rec in resume), encoding="utf-8")
        if self.best is not None:
            self.keep_best()

    @staticmethod
    def key(params):
        """A point's cache key: its values as the log keeps them (LOG_DECIMALS), so that a resumed log's points match."""
        return tuple(sorted((k, round(v, LOG_DECIMALS) if isinstance(v, float) else v) for k, v in params.items()))

    def known(self, params):
        """Whether `params` (cleaned) were measured already: evaluating them costs nothing."""
        return self.key(self.clean(params)) in self.cache

    def keep_best(self):
        (self.out / "best.json").write_text(json.dumps(jsonx.to_jsonable(self.best, LOG_DECIMALS), indent=1),
                                            encoding="utf-8")
        self.params_file(self.best["params"], "best_params.json")

    def clean(self, params):
        """Params clamped into their bounds, whole numbers rounded."""
        out = {}
        for k, v in params.items():
            p = self.space[k]
            if "choices" not in p:
                v = float(v)
                if p["min"] is not None:
                    v = max(float(p["min"]), v)
                if p["max"] is not None:
                    v = min(float(p["max"]), v)
                v = round(v) if p["int"] else round(v, 9)
            out[k] = v
        return out

    def params_file(self, params, name="params.json"):
        path = self.out / name
        path.write_text(json.dumps({**self.fixed, **params}, indent=1), encoding="utf-8")
        return path

    def __call__(self, params):
        params = self.clean(params)
        key = self.key(params)
        if key in self.cache:
            return self.cache[key]
        self.n += 1
        t0 = time.time()
        rec = {"step": self.done + self.n, "params": params, "metric": self.metric, "pool": bool(self.pool),
               **({"weights": self.weights} if self.weights else {})}
        try:
            frames = self.evaluate(self.params_file(params))
            if not frames:
                raise ParityError("no frames were measured")
            per = {}
            for part, _, m in frames:
                per.setdefault(part, []).append((score(m, self.metric), float(m.get("pixels") or 1)))
            # a part's score: its frames' mean, or (pool) their pixels' mean
            parts = {k: float(np.average([s for s, _ in v], weights=[n for _, n in v]) if self.pool
                              else np.mean([s for s, _ in v])) for k, v in per.items()}
            groups = {}
            for k, v in per.items():
                # a group's score: the mean over its frames, or (pool) over its parts
                groups.setdefault(k.split(":", 1)[0] if ":" in k else "", []).extend([parts[k]] if self.pool else
                                                                                      [s for s, _ in v])
            value = float(sum(self.weights.get(k, 1.0) * np.mean(v) for k, v in groups.items()))
            rec.update(score=value, parts=parts)
            if len(groups) > 1:
                rec["groups"] = {k: float(np.mean(v)) for k, v in groups.items()}
        except Exception as e:  # noqa: BLE001 - mkr rejecting a value, a crash: the point is just bad
            value = math.inf
            rec.update(score=None, error=f"{type(e).__name__}: {e}"[:600])
        rec["seconds"] = round(time.time() - t0, 2)
        rec["best"] = value < (self.best["score"] if self.best else math.inf)
        if self.first is None:
            self.first = rec
        if rec["best"]:
            self.best = dict(rec, score=value)
            self.keep_best()
        with open(self.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(jsonx.to_jsonable(rec, LOG_DECIMALS)) + "\n")
        best = self.best["score"] if self.best else math.inf
        shown = " ".join(f"{k}={v}" for k, v in params.items())
        self.say(f"tune {self.done + self.n}: {self.metric} {value:.4f} (best {best:.4f}) {shown}" +
                 (f" [{rec['error']}]" if "error" in rec else ""))
        self.cache[key] = value
        return value


def coordinate(obj, x0, budget):
    """Coordinate descent: each number one step up and down (going on while it pays), each choice tried; a sweep
    that moves nothing halves the steps (whole numbers down to 1, others down to min_step, default step / 64); stops
    there or after `budget` evaluations."""
    space = obj.space
    x, fx = dict(x0), obj(x0)
    steps = {k: p["step"] for k, p in space.items() if "choices" not in p}
    floor = {k: 1 if space[k]["int"] else (space[k].get("min_step") or steps[k] / 64) for k in steps}
    while obj.n < budget:
        moved = False
        for k, p in space.items():
            if obj.n >= budget:
                break
            if "choices" in p:
                for c in p["choices"]:
                    if c != x[k] and obj.n < budget:
                        y = dict(x, **{k: c})
                        fy = obj(y)
                        if fy < fx:
                            x, fx, moved = y, fy, True
                continue
            for sign in (1, -1):
                y = obj.clean(dict(x, **{k: x[k] + sign * steps[k]}))
                if y[k] == x[k] or obj.n >= budget:
                    continue
                fy = obj(y)
                if fy >= fx:
                    continue
                x, fx, moved = y, fy, True
                while obj.n < budget:
                    y = obj.clean(dict(x, **{k: x[k] + sign * steps[k]}))
                    if y[k] == x[k]:
                        break
                    fy = obj(y)
                    if fy >= fx:
                        break
                    x, fx = y, fy
                break
        if not moved:
            shrunk = False
            for k, s in steps.items():
                if s > floor[k]:
                    steps[k] = max(1, int(s) // 2) if space[k]["int"] else max(floor[k], s / 2)
                    shrunk = True
            if not shrunk:
                break
    return x, fx


class _Spent(Exception):
    """The budget of new evaluations is spent: nelder_mead's way out of scipy."""


def nelder_mead(obj, x0, budget, min_step=None):
    """Nelder-Mead (scipy) over the numeric params from a simplex of one step along each, within their bounds; stops
    after `budget` new evaluations (points measured already cost nothing)."""
    from scipy.optimize import minimize
    space = obj.space
    names = [k for k in space if "choices" not in space[k]]
    if len(names) != len(space):
        raise ParityError("nelder-mead tunes numbers only: use --method coord for params with choices")
    v0 = np.array([float(x0[k]) for k in names])
    simplex = np.array([v0] + [v0 + np.eye(len(names))[i] * space[k]["step"] for i, k in enumerate(names)])
    bounds = [(space[k]["min"], space[k]["max"]) for k in names]

    def f(v):
        params = dict(zip(names, v.tolist()))
        if obj.n >= budget and not obj.known(params):
            raise _Spent
        return obj(params)
    try:
        minimize(f, v0, method="Nelder-Mead", bounds=None if all(b == (None, None) for b in bounds) else bounds,
                 options={"maxfev": 10 * (budget + len(obj.cache)) + 200 * len(names), "initial_simplex": simplex,
                          "xatol": min_step or 1e-4, "fatol": 1e-4, "adaptive": len(names) > 2})
    except _Spent:
        pass
    return obj.best["params"], obj.best["score"]


def read_log(path):
    """An earlier tune's evaluations (its OUT/tune.jsonl), for `tune(resume=)`; a last line cut short (the run killed
    while writing it) is left out."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise ParityError(f"{path}: {e.strerror or e}") from None
    lines = text.splitlines()
    recs = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            if i == len(lines) and not text.endswith("\n"):
                break
            raise ParityError(f"{path}:{i}: not a line of a tune log") from None
        if not isinstance(rec, dict) or not isinstance(rec.get("params"), dict):
            raise ParityError(f"{path}:{i}: not a line of a tune log")
        recs.append(rec)
    return recs


def tune(evaluate, tspec, out, say=None, defaults=None, method=None, iterations=None, metric=None, pool=None,
         resume=()):
    """Hill-climb mkr's tunables (`tspec`: load_space's) on `evaluate`; the best params, its score, the starting
    point's and the log's paths. `defaults`: mkr's defaults (`mkr params`) for params without a start; `metric` and
    `pool` override the file's. `resume`: an earlier run's log (`read_log`) of the same tune file and options: the same
    search walks through its points again for nothing and goes on from there, for `iterations` new evaluations."""
    space = tspec["params"]
    for k, p in space.items():
        if p["start"] is None:
            if defaults is None or k not in defaults:
                raise ParityError(f"{k}: no start in the tune file and mkr gives no default for it")
            p["start"] = defaults[k]
        if tspec.get("min_step") is not None and "choices" not in p:
            p["min_step"] = float(tspec["min_step"])
    method = method or tspec["method"]
    budget = int(iterations or tspec["iterations"])
    metric = metric or tspec["metric"]
    pool = tspec.get("pool", False) if pool is None else pool
    if pool and metric not in POOLED:
        raise ParityError(f"pool scores {', '.join(POOLED)} only, not {metric!r}")
    obj = Objective(evaluate, metric, space, tspec["fixed"], out, say, pool=pool, resume=resume,
                    weights=tspec.get("weights"))
    x0 = obj.clean({k: p["start"] for k, p in space.items()})
    t0 = time.time()
    if method == "coord":
        coordinate(obj, x0, budget)
    elif method == "nelder-mead":
        obj(x0)
        nelder_mead(obj, x0, budget, tspec.get("min_step"))
    else:
        raise ParityError(f"unknown method {method!r} (coord, nelder-mead)")
    if obj.best is None or not math.isfinite(obj.best["score"]):
        raise ParityError(f"no evaluation succeeded (see {obj.log}): {(obj.first or {}).get('error')}")
    return {"metric": obj.metric, "pool": pool, **({"weights": obj.weights} if obj.weights else {}), "method": method,
            "evaluations": obj.n, "resumed": len(resume),
            "seconds": round(time.time() - t0, 1),
            "start": {"params": obj.first["params"], "score": obj.first.get("score")},
            "best": {"params": obj.best["params"], "score": obj.best["score"], "step": obj.best["step"],
                     "parts": obj.best.get("parts"), **({"groups": obj.best["groups"]} if "groups" in obj.best else {})},
            "log": str(obj.log), "best_params": str(obj.out / "best_params.json")}
