"""mk parity (mkmmd/render/parity.py, mkmmd/render/lab.py, docs/design.md: Parity): the metric maths on frames with known
errors, folders matched by name and pooled, frames of a look left out, hill climbing, and `frames`, `lab` and `tune`
end to end with a stand-in for mkr (a script that copies Blender's frames shifted by 40 * (exposure - 0.3) levels)."""
import json
import math
import stat
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from mkmmd.cli import main as MAIN
from mkmmd.render import parity as PY


def png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, np.uint8)).save(path)
    return path


def flat(h, w, rgb):
    return np.broadcast_to(np.asarray(rgb, np.uint8), (h, w, 3)).copy()


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


# ================================================================================================= metric maths
def test_identical_frames_have_no_error():
    a = np.random.default_rng(1).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    m = PY.metrics(PY.stats(a, a.copy()))
    assert m["pixels"] == 2400
    assert m["mean_abs"] == 0 and m["bias"] == [0, 0, 0] and m["over8"] == 0 and m["over32"] == 0
    assert m["p99"] == 0 and m["max"] == 0 and m["rmse"] == 0 and m["psnr"] is None
    assert PY.passes(m)


def test_a_constant_offset_in_one_channel():
    ref = flat(20, 30, (100, 100, 100))
    test = flat(20, 30, (103, 100, 100))
    m = PY.metrics(PY.stats(ref, test))
    assert m["mean_abs"] == pytest.approx(1.0)                       # 3 levels in one channel of three
    assert m["bias"] == pytest.approx([3.0, 0.0, 0.0])
    assert m["over8"] == 0 and m["p99"] == 3 and m["max"] == 3
    assert m["rmse"] == pytest.approx(math.sqrt(3.0))
    assert m["psnr"] == pytest.approx(10 * math.log10(255 ** 2 / 3.0))
    darker = PY.metrics(PY.stats(test, ref))
    assert darker["bias"] == pytest.approx([-3.0, 0.0, 0.0])          # test - ref: the test is darker


def test_pixels_over_8_and_32_the_p99_and_the_max():
    ref = flat(100, 100, (50, 60, 70))
    test = ref.copy()
    test.reshape(-1, 3)[:150, 1] += 20                                # 1.5 % of pixels off by 20 (green)
    test.reshape(-1, 3)[150:250, 2] += 40                             # 1 % off by 40 (blue)
    m = PY.metrics(PY.stats(ref, test))
    assert m["over8"] == pytest.approx(2.5) and m["over32"] == pytest.approx(1.0)
    assert m["p99"] == 20 and m["max"] == 40                          # 97.5 % at 0, 99 % within 20
    assert m["mean_abs"] == pytest.approx((150 * 20 + 100 * 40) / 30000)
    assert m["bias"] == pytest.approx([0.0, 0.3, 0.4])
    assert not PY.passes(m)                                           # 2.5 % > 8 levels is not 1:1


def test_a_mask_and_a_region_limit_the_pixels(tmp_path):
    ref = flat(10, 20, (0, 0, 0))
    test = ref.copy()
    test[:, 10:] = 60                                                 # the right half is wrong
    left = PY.mask_for(1, (10, 20), region=(0, 0, 0.5, 1))           # fractions of the frame
    assert left.sum() == 100 and PY.metrics(PY.stats(ref, test, left))["mean_abs"] == 0
    right = PY.mask_for(1, (10, 20), region=(10, 0, 20, 10))         # pixels
    assert PY.metrics(PY.stats(ref, test, right))["mean_abs"] == 60
    m = np.zeros((10, 20), np.uint8)
    m[:, 15:] = 255
    mask = PY.mask_for(1, (10, 20), mask=png(tmp_path / "m.png", np.repeat(m[:, :, None], 3, 2)), region=(0, 0, 1, 1))
    assert mask.sum() == 50 and PY.metrics(PY.stats(ref, test, mask))["over32"] == 100


def test_16_bit_frames_count_in_8_bit_levels(tmp_path):
    ref = np.full((8, 8, 3), 257 * 100, np.uint16)
    test = ref.copy()
    test[..., 0] += 257                                              # one 8-bit level in red
    test[..., 1] += 128                                              # half a level in green
    for name, a in (("r.png", ref), ("t.png", test)):
        cv2.imwrite(str(tmp_path / name), a[:, :, ::-1])
    r, t = PY.read(tmp_path / "r.png"), PY.read(tmp_path / "t.png")
    assert r.dtype == np.float32 and float(r[0, 0, 0]) == pytest.approx(100.0)
    m = PY.metrics(PY.stats(r, t))
    assert m["bias"] == pytest.approx([1.0, 128 * 255 / 65535, 0.0], abs=1e-4)
    eight = PY.metrics(PY.stats(PY.read(png(tmp_path / "e.png", flat(8, 8, (101, 100, 100)))), r))
    assert eight["bias"] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-4)  # 8-bit against 16-bit, in one unit


def test_the_figure_leaves_the_black_void_out(tmp_path):
    ref = flat(10, 20, (0, 0, 0))
    ref[2:8, 5:15] = 100                                             # a figure of 60 pixels on black
    test = ref.copy()
    test[2:8, 5:15] = 104                                            # 4 levels off on the figure
    test[0, 0] = 9                                                   # a stray pixel in the void counts too
    png(tmp_path / "r" / "00001.png", ref)
    png(tmp_path / "t" / "00001.png", test)
    whole = PY.compare(tmp_path / "r", tmp_path / "t")["overall"]
    fig = PY.compare(tmp_path / "r", tmp_path / "t", figure=4.0)["overall"]
    assert whole["pixels"] == 200 and whole["mean_abs"] == pytest.approx((60 * 12 + 27) / 600)
    assert fig["pixels"] == 61 and fig["mean_abs"] == pytest.approx((60 * 12 + 27) / 183)


def test_compare_matches_frames_by_name_and_pools_their_pixels(tmp_path):
    ref, test = tmp_path / "ref", tmp_path / "test"
    png(ref / "00091.png", flat(10, 10, (10, 10, 10)))
    png(ref / "00092.png", flat(20, 20, (10, 10, 10)))
    png(ref / "00093.png", flat(10, 10, (10, 10, 10)))
    png(test / "91.png", flat(10, 10, (13, 13, 13)))                # off by 3 on 100 pixels
    png(test / "92.png", flat(20, 20, (22, 22, 22)))                # off by 12 on 400 pixels
    (test / "00094.png").touch()                                      # an empty claim of a render in progress
    s = PY.compare(ref, test, tmp_path / "out", signed=True)
    assert s["frames"] == 2 and s["only_ref"] == "93" and s["only_test"] == ""
    assert s["per_frame"][91]["mean_abs"] == 3 and s["per_frame"][92]["mean_abs"] == 12
    assert s["overall"]["mean_abs"] == pytest.approx((100 * 3 + 400 * 12) / 500)   # pooled, not the mean of frames
    assert s["overall"]["over8"] == pytest.approx(80.0)
    assert [w["frame"] for w in s["worst"]] == [92, 91]
    saved = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert saved["overall"]["mean_abs"] == pytest.approx(10.2) and set(saved["per_frame"]) == {"91", "92"}
    sheet = np.asarray(Image.open(s["sheets"][0]))
    assert sheet.shape[1] == 4 * 20 + 3 * 6                          # four panels across (signed), frames at full size
    assert PY.compare(ref, test, None, frames=[92])["frames"] == 1


def test_frames_of_different_sizes_are_errors_not_metrics(tmp_path):
    png(tmp_path / "ref" / "00001.png", flat(10, 10, (0, 0, 0)))
    png(tmp_path / "test" / "00001.png", flat(10, 12, (0, 0, 0)))
    s = PY.compare(tmp_path / "ref", tmp_path / "test")
    assert s["frames"] == 0 and "sizes differ" in s["errors"][1]


def test_frames_of_a_look_a_medium_or_a_fisheye_are_left_out():
    doc = {"shots": [{"name": "a", "from": 1, "to": 5, "cameras": {"main": "a@main", "sq": "a@sq"}},
                     {"name": "b", "from": 5, "to": 8, "cameras": {"main": "b@main", "sq": "b@sq"},
                      "styles": {"main": {"silhouette": {}}}},
                     {"name": "p", "from": 2, "to": 9, "plate": True, "cameras": {"main": "p@main"},
                      "styles": {"main": {"vector": {}}}},
                     {"name": "c", "from": 8, "to": 12, "cameras": {"main": "c@main", "sq": "c@sq"}}],
           "cameras": [{"name": "c@sq", "fisheye": {"projection": "equisolid"}}],
           "medium": [{"index": 0, "from": 2, "to": 3, "styles": {"main": {"vector": {}}, "sq": {}}}]}
    main = PY.looked(doc, "main", range(1, 12))
    assert sorted(main) == [2, 5, 6, 7]
    assert main[2] == "medium 0: a vector look" and main[5] == "shot b: a silhouette look"
    sq = PY.looked(doc, "sq", range(1, 12))
    assert sorted(sq) == [8, 9, 10, 11] and "fisheye" in sq[8]


# ================================================================================================= hill climbing
def _quadratic(target, log):
    """An evaluate whose frames' mean |error| is the squared distance of the params from `target`."""
    def evaluate(params_file, draw=False):
        p = json.loads(Path(params_file).read_text())
        log.append(p)
        if p.get("x", 0) > 5:
            raise PY.ParityError("mkr refuses x > 5")
        err = sum((p[k] - v) ** 2 for k, v in target.items())
        return [("case/main", 1, {"mean_abs": err}), ("case/main", 2, {"mean_abs": err})]
    return evaluate


def test_coordinate_descent_finds_the_minimum_and_logs_every_step(tmp_path):
    seen = []
    tspec = {"params": {"x": {"start": 1.0, "step": 0.5, "min": -2.0, "max": 8.0, "int": False},
                        "n": {"start": 1, "step": 4, "min": 0, "max": 20, "int": True},
                        "f": {"choices": ["box", "gauss"], "start": "box"}},
             "fixed": {"keep": 7}, "metric": "mean_abs", "method": "coord", "iterations": 200, "min_step": 0.01}

    def evaluate(params_file, draw=False):
        p = json.loads(Path(params_file).read_text())
        seen.append(p)
        err = (p["x"] - 0.3) ** 2 + (p["n"] - 6) ** 2 + (0.0 if p["f"] == "gauss" else 1.0)
        return [("c/main", 1, {"mean_abs": err})]
    res = PY.tune(evaluate, tspec, tmp_path / "t")
    assert res["best"]["params"]["n"] == 6 and res["best"]["params"]["f"] == "gauss"
    assert res["best"]["params"]["x"] == pytest.approx(0.3, abs=0.01) and res["best"]["score"] < 1e-3
    assert res["start"]["score"] == pytest.approx(0.49 + 25 + 1)
    lines = (tmp_path / "t" / "tune.jsonl").read_text().splitlines()
    assert len(lines) == res["evaluations"] == len(seen)              # every evaluation logged, no point twice
    assert len({json.dumps(p, sort_keys=True) for p in seen}) == len(seen)
    best = json.loads((tmp_path / "t" / "best_params.json").read_text())
    assert best == {"keep": 7, **res["best"]["params"]}               # mkr's --params file: fixed plus the best


def test_nelder_mead_and_a_render_that_fails(tmp_path):
    seen = []
    tspec = PY.load_space(_write(tmp_path / "t.json", {"params": {"x": {"start": 4.0, "step": 1.0},
                                                                  "y": {"start": 0.0, "step": 1.0, "max": 3.0}},
                                                       "method": "nelder-mead", "iterations": 120}))
    res = PY.tune(_quadratic({"x": 1.5, "y": -1.0}, seen), tspec, tmp_path / "o")
    assert res["best"]["params"]["x"] == pytest.approx(1.5, abs=0.02)
    assert res["best"]["params"]["y"] == pytest.approx(-1.0, abs=0.02)
    failed = [json.loads(line) for line in (tmp_path / "o" / "tune.jsonl").read_text().splitlines()
              if "error" in json.loads(line)]
    assert all(r["params"]["x"] > 5 and r["score"] is None for r in failed)   # x > 5 scored infinity, logged


def test_a_start_comes_from_mkrs_defaults(tmp_path):
    tspec = PY.load_space(_write(tmp_path / "t.json", {"params": {"x": {"step": 0.25}}, "iterations": 40}))
    res = PY.tune(_quadratic({"x": 2.0}, []), tspec, tmp_path / "o", defaults={"x": 1.0})
    assert res["start"]["params"] == {"x": 1.0} and res["best"]["params"]["x"] == pytest.approx(2.0)
    with pytest.raises(PY.ParityError):
        PY.tune(_quadratic({"x": 2.0}, []), PY.load_space(tmp_path / "t.json"), tmp_path / "p", defaults={})


def test_pool_weighs_a_parts_frames_by_their_pixels_and_its_parts_alike(tmp_path):
    def evaluate(params_file, draw=False):
        x = json.loads(Path(params_file).read_text())["x"]
        # part a: a big frame 1 level off and a small one 9 off; part b: one frame |x - 2| off
        return [("a", 1, {"mean_abs": 1.0, "pixels": 900}), ("a", 2, {"mean_abs": 9.0, "pixels": 100}),
                ("b", 1, {"mean_abs": abs(x - 2.0), "pixels": 50})]
    tspec = {"params": {"x": {"start": 0.0, "step": 1.0, "min": 0.0, "max": 4.0, "int": False}}, "fixed": {},
             "metric": "mean_abs", "method": "coord", "iterations": 30, "min_step": 0.25, "pool": True}
    res = PY.tune(evaluate, tspec, tmp_path / "t")
    assert res["pool"] and res["start"]["score"] == pytest.approx((1.8 + 2.0) / 2)   # a pooled: 1.8; b: 2
    assert res["best"]["params"]["x"] == pytest.approx(2.0) and res["best"]["parts"]["a"] == pytest.approx(1.8)
    unpooled = PY.tune(evaluate, dict(tspec, pool=False), tmp_path / "u")
    assert unpooled["start"]["score"] == pytest.approx((1.0 + 9.0 + 2.0) / 3)          # every frame alike
    with pytest.raises(PY.ParityError):
        PY.tune(evaluate, dict(tspec, metric="p99"), tmp_path / "v")


@pytest.mark.parametrize("method", ["coord", "nelder-mead"])
def test_a_tune_resumed_in_turns_walks_the_same_way_as_one_run(tmp_path, method):
    def spec():
        return {"params": {"x": {"start": 1.0, "step": 0.5, "min": -2.0, "max": 8.0, "int": False},
                           "y": {"start": 0.0, "step": 0.75, "min": None, "max": None, "int": False}},
                "fixed": {}, "metric": "mean_abs", "method": method, "iterations": 40, "min_step": 0.01}
    target = {"x": 0.3, "y": -1.15}
    whole = []
    one = PY.tune(_quadratic(target, whole), spec(), tmp_path / "one")
    turns, out = [], tmp_path / "turns"
    res = PY.tune(_quadratic(target, turns), spec(), out, iterations=5)
    assert res["evaluations"] == 5 and res["resumed"] == 0
    for _ in range(7):                                          # 8 turns of 5: the one run's 40
        res = PY.tune(_quadratic(target, turns), spec(), out, iterations=5, resume=PY.read_log(out / "tune.jsonl"))
        assert res["evaluations"] <= 5
    assert turns == whole                                       # the same points in the same order, none twice
    assert res["best"]["params"] == pytest.approx(one["best"]["params"], abs=1e-6)
    assert res["start"]["score"] == pytest.approx(one["start"]["score"], abs=1e-6)
    log = [json.loads(line) for line in (out / "tune.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == list(range(1, len(whole) + 1))
    with pytest.raises(PY.ParityError):                         # another metric's log is no start
        PY.tune(_quadratic(target, []), dict(spec(), metric="rmse"), tmp_path / "x", resume=log)
    cut = tmp_path / "cut.jsonl"                                # a run killed while writing its last line
    cut.write_text("".join(json.dumps(r) + "\n" for r in log[:3]) + json.dumps(log[3])[:20])
    assert PY.read_log(cut) == log[:3]


def test_weights_scale_a_groups_mean(tmp_path):
    def evaluate(params_file, draw=False):
        x = json.loads(Path(params_file).read_text())["x"]
        # the project wants x = 3, the lab x = 1: their weights decide where the best lies
        return [("project:a", 1, {"mean_abs": abs(x - 3.0)}), ("lab:c/main", 1, {"mean_abs": abs(x - 1.0)}),
                ("lab:d/main", 1, {"mean_abs": abs(x - 1.0)})]
    tspec = PY.load_space(_write(tmp_path / "t.json", {"params": {"x": {"start": 2.0, "step": 0.5, "min": 0, "max": 4}},
                                                       "iterations": 30, "min_step": 0.25, "weights": {"lab": 3}}))
    res = PY.tune(evaluate, tspec, tmp_path / "w")
    assert res["weights"] == {"lab": 3.0} and res["start"]["score"] == pytest.approx(1.0 + 3 * 1.0)
    assert res["best"]["params"]["x"] == pytest.approx(1.0) and res["best"]["groups"] == {"project": 2.0, "lab": 0.0}
    plain = PY.tune(evaluate, dict(tspec, weights={}), tmp_path / "p")     # alike: anywhere in [1, 3] scores 2
    assert plain["best"]["score"] == pytest.approx(2.0)
    with pytest.raises(PY.ParityError):                         # a log weighed otherwise is no start
        PY.tune(evaluate, dict(tspec, weights={}), tmp_path / "q", resume=PY.read_log(tmp_path / "w" / "tune.jsonl"))
    with pytest.raises(PY.ParityError):
        PY.load_space(_write(tmp_path / "bad.json", {"params": {"x": {"step": 1}}, "weights": {"lab": -1}}))


def _write(path, doc):
    path.write_text(json.dumps(doc))
    return path


# ================================================================================================= with a stand-in mkr
FAKE = '''#!{python}
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
with open(os.environ["FAKE_MKR_LOG"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
with open(os.environ["FAKE_MKR_HELD"], "a") as fh:            # whether its caller holds the GPU lock
    fh.write(os.environ.get("MK_GPU_LOCK_HELD", "-") + "\\n")
p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd")
r = sub.add_parser("render")
for k in ("--bake", "--output", "--frames", "--out", "--samples", "--percent", "--shutter", "--params"):
    r.add_argument(k)
r.add_argument("--motion-blur", action="store_true")
sub.add_parser("params")
a = p.parse_args()
if a.cmd == "params":
    print(json.dumps({{"exposure": 0.0}}))
    sys.exit(0)
refs = Path(os.environ.get("FAKE_MKR_REFS") or Path(a.bake).parent / "ref") / a.output
off = 40.0 * (json.loads(Path(a.params).read_text()).get("exposure", 0.0) - 0.3) if a.params else 0.0
frames = []
for part in a.frames.split(","):
    lo, _, hi = part.partition(":")
    frames += list(range(int(lo), int(hi or lo) + 1))
for f in frames:
    im = np.asarray(Image.open(refs / f"{{f:05d}}.png").convert("RGB")).astype(np.float32)
    Image.fromarray(np.clip(np.rint(im + off), 0, 255).astype(np.uint8)).save(Path(a.out) / f"{{f:05d}}.png")
print(json.dumps({{"rendered": len(frames), "seconds": 0.01, "s_per_frame": 0.01, "gpu_ms_per_frame": 0.0,
                  "adapter": "fake", "warnings": []}}))
'''


@pytest.fixture
def mkr(tmp_path, monkeypatch):
    path = tmp_path / "fake_mkr"
    path.write_text(FAKE.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.delenv("FAKE_MKR_REFS", raising=False)
    log = tmp_path / "mkr_calls.jsonl"
    monkeypatch.setenv("MK_MKR", str(path))
    monkeypatch.setenv("FAKE_MKR_LOG", str(log))
    monkeypatch.setenv("FAKE_MKR_HELD", str(tmp_path / "mkr_held.txt"))
    monkeypatch.setenv("MK_GPU_LOCK", str(tmp_path / "gpu.lock"))      # never the machine's own GPU lock
    monkeypatch.delenv("MK_GPU_LOCK_HELD", raising=False)
    return lambda: [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _project(root):
    """A project of one output whose shot b (frames 5-6) is a silhouette, with Blender's final frames 1-8."""
    root.mkdir(parents=True)
    (root / "mk.toml").write_text('[project]\nname = "p"\nfps = 30\nframe0 = 1\nduration = 0.3\nblend = "build/p.blend"\n'
                                  '[render]\nshutter = 0.5\n[[output]]\nname = "main"\nsize = [32, 18]\n')
    bake = root / "build" / "p.bake"
    bake.mkdir(parents=True)
    (bake / "bake.json").write_text(json.dumps({
        "version": 5, "frame0": 1, "frames": 9, "outputs": [{"name": "main", "size": [32, 18]}],
        "cameras": [], "shots": [{"name": "a", "from": 1, "to": 5, "cameras": {"main": "a@main"}},
                                 {"name": "b", "from": 5, "to": 7, "cameras": {"main": "b@main"},
                                  "styles": {"main": {"silhouette": {}}}},
                                 {"name": "c", "from": 7, "to": 10, "cameras": {"main": "c@main"}}]}))
    rng = np.random.default_rng(3)
    for f in range(1, 9):
        png(root / "renders" / "final" / "main" / f"{f:05d}.png", rng.integers(60, 200, (18, 32, 3)))
    return root


def test_frames_renders_the_lit_frames_with_mkr_and_compares(tmp_path, capsys, mkr, monkeypatch):
    proj = _project(tmp_path / "p")
    monkeypatch.setenv("FAKE_MKR_REFS", str(proj / "renders" / "final"))
    params = _write(tmp_path / "params.json", {"exposure": 0.4})
    code, out = cli(["parity", "frames", "--project", str(proj), "--run", "r1", "--params", str(params)], capsys)
    assert code == 0
    o = out["outputs"]["main"]
    assert o["skipped"] == {"shot b: a silhouette look": "5:6"} and o["drawn"] == "1:4,7:8"
    assert o["frames"] == 6 and o["percent"] == 100
    assert o["overall"]["mean_abs"] == pytest.approx(4.0) and o["overall"]["bias"] == pytest.approx([4.0] * 3)
    call = mkr()[0]
    assert call[:2] == ["render", "--bake"] and call[call.index("--frames") + 1] == "1:4,7:8"
    assert call[call.index("--motion-blur") + 2] == "0.5"                # final: motion blur, [render] shutter
    # final at its own samples: mk render's mkr sampling, the given tunables over its own (exposure: 4 levels off)
    assert call[call.index("--samples") + 1] == "36" and call.count("--params") == 1
    run = proj / ".mk" / "parity" / "r1"
    assert (run / "main" / "frames" / "00007.png").exists() and (run / "main" / "compare" / "summary.json").exists()
    assert json.loads((run / "run.json").read_text())["overall"]["mean_abs"] == pytest.approx(4.0)
    code, _ = cli(["parity", "frames", "--project", str(proj), "--gate", "--no-sheet", "--no-motion-blur", "--params",
                   str(params)], capsys)
    assert code == 1 and "--motion-blur" not in mkr()[1]               # 4 levels off is not 1:1
    cli(["parity", "frames", "--project", str(proj), "--no-sheet", "--samples", "64", "--params", str(params)], capsys)
    call = mkr()[2]                                                     # samples asked for: as asked, the file as given
    assert call[call.index("--samples") + 1] == "64" and call[call.index("--params") + 1] == str(params)


def _lab_case(root, name="area_square"):
    """A built lab case as the lab keeps one (frames 1 and 3 of three are Blender's), without Blender."""
    d = root / name
    case = {"name": name, "kind": "light", "doc": "", "frames": 3, "refs": [1, 3], "labels": ["one", "three"],
            "outputs": [{"name": "main", "size": [24, 16]}], "samples": 64, "motion_blur": 0.35}
    (d / "scene.bake").mkdir(parents=True)
    (d / "scene.blend").write_bytes(b"")
    (d / "case.json").write_text(json.dumps({"case": case, "report": {"bake": {"version": 5}}}))
    for f in (1, 3):
        png(d / "ref" / "main" / f"{f:05d}.png", np.full((16, 24, 3), 100 + 40 * f))
    return d


def test_lab_draws_each_case_with_mkr_and_prints_the_table(tmp_path, capsys, mkr):
    _lab_case(tmp_path / "lab")
    code, out = cli(["parity", "lab", "area_*", "--root", str(tmp_path / "lab"), "--params",
                     str(_write(tmp_path / "p.json", {"exposure": 0.3})), "--per-frame"], capsys)
    assert code == 0
    rows = {(r["frame"], r["label"]): r for r in out["rows"]}
    assert set(rows) == {(None, None), (1, "one"), (3, "three")} and rows[(None, None)]["mean_abs"] == 0
    assert out["notes"] == {n: "not built: `mk parity lab --build`" for n in ("area_rect", "area_disk", "area_ellipse")}
    call = mkr()[0]
    assert call[call.index("--frames") + 1] == "1,3" and call[call.index("--shutter") + 1] == "0.35"
    assert (tmp_path / "lab" / "area_square" / "compare" / "main" / "sheet_01.png").exists()
    code, out = cli(["parity", "lab", "nope", "--root", str(tmp_path / "lab")], capsys)
    assert code == 2 and "no lab case" in out["error"]


def test_tune_climbs_mkrs_tunables_on_lab_cases(tmp_path, capsys, mkr):
    lab = tmp_path / "lab"
    _lab_case(lab)
    spec = _write(tmp_path / "tune.json", {"params": {"exposure": {"step": 0.2, "min": -1, "max": 1}},
                                           "fixed": {"other": 1}, "iterations": 25})
    code, out = cli(["parity", "tune", str(spec), "--cases", "area_square", "--root", str(lab),
                     "--out", str(tmp_path / "t")], capsys)
    assert code == 0
    assert out["start"]["params"] == {"exposure": 0.0}                # mkr's default (`mkr params`)
    assert out["best"]["params"]["exposure"] == pytest.approx(0.3, abs=0.0125) and out["best"]["score"] == 0
    calls = [c for c in mkr() if c[0] == "render"]
    assert len(calls) == out["evaluations"] + 1                        # every step, and the best once more for its sheet
    held = (tmp_path / "mkr_held.txt").read_text().split()
    assert [h for c, h in zip(mkr(), held) if c[0] == "render"] == ["1"] * len(calls)   # each evaluation holds the GPU
    assert json.loads((tmp_path / "t" / "best_params.json").read_text())["other"] == 1
    assert (tmp_path / "t" / "scratch" / "area_square" / "compare" / "main" / "sheet_01.png").exists()


def test_tune_on_a_project_and_the_lab_sums_their_means(tmp_path, capsys, mkr):
    lab = tmp_path / "lab"
    _lab_case(lab)
    proj = _project(tmp_path / "p")
    # the stand-in mkr finds Blender's frames beside the bake it is given: the lab case's ref/, the project's here
    for f in range(1, 9):
        src = proj / "renders" / "final" / "main" / f"{f:05d}.png"
        png(proj / "build" / "ref" / "main" / f"{f:05d}.png", np.asarray(Image.open(src)))
    spec = _write(tmp_path / "tune.json", {"params": {"exposure": {"step": 0.2, "min": -1, "max": 1}}, "iterations": 25})
    code, out = cli(["parity", "tune", str(spec), "--project", str(proj), "--cases", "area_square", "--root", str(lab),
                     "--out", str(tmp_path / "t"), "--no-motion-blur"], capsys)
    assert code == 0, out
    assert out["start"]["score"] == pytest.approx(12.0 + 12.0)          # 40 x 0.3 levels off on either side
    assert out["best"]["params"]["exposure"] == pytest.approx(0.3, abs=0.0125) and out["best"]["score"] == 0
    assert set(out["best"]["groups"]) == {"project", "lab"}
    assert set(out["best"]["parts"]) == {"project:main", "lab:area_square/main"}
    first = json.loads((tmp_path / "t" / "tune.jsonl").read_text().splitlines()[0])
    assert first["groups"] == {"project": pytest.approx(12.0), "lab": pytest.approx(12.0)}
    assert (tmp_path / "t" / "scratch" / "lab" / "area_square" / "compare" / "main" / "sheet_01.png").exists()
    assert (tmp_path / "t" / "scratch" / "project" / "main" / "compare").is_dir()


def test_compare_cli_gates_on_the_1_to_1_bar(tmp_path, capsys):
    png(tmp_path / "a" / "00001.png", flat(8, 8, (10, 10, 10)))
    png(tmp_path / "b" / "00001.png", flat(8, 8, (11, 10, 10)))
    code, out = cli(["parity", "compare", str(tmp_path / "a"), str(tmp_path / "b"), "--gate"], capsys)
    assert code == 0 and out["pass"] and out["overall"]["mean_abs"] == pytest.approx(1 / 3)
    assert (tmp_path / "b-compare" / "summary.json").exists()
    png(tmp_path / "c" / "00001.png", flat(8, 8, (40, 10, 10)))
    code, out = cli(["parity", "compare", str(tmp_path / "a"), str(tmp_path / "c"), "--gate", "--no-sheet"], capsys)
    assert code == 1 and not out["pass"] and "sheets" not in out
    code, out = cli(["parity", "compare", str(tmp_path / "a"), str(tmp_path / "c"), "--frames", "5"], capsys)
    assert code == 2 and "no frame" in out["error"]
