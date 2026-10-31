"""The parity lab (mkmmd/render/lab.py, mkmmd/blender/ops_lab.py, docs/design.md: Parity): with Blender, a case becomes a
tiny project with its scene, Blender's frames in three forms and its bake, and the frames are what the case says (the
view transform's ramp through Standard is the sRGB curve of its scene-linear values); a variant renders the saved scene
again (shadows off, Cycles) and leaves the case alone. Without Blender, on cases drawn by hand and a stand-in for mkr
(Blender's frames copied, shifted by 40 * (exposure - 0.3) levels, the unshadowed ones with {"shadows": false}):
`--against` measures mkr against a variant and `--separate` splits the error into its parts."""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN
from mkmmd.render import lab as LB
from mkmmd.render import parity as PY


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


needs_blender = pytest.mark.skipif(not have_blender(), reason="Blender not available")


def srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


def png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, np.uint8)).save(path)
    return path


def digest(d):
    """{relative path: sha256} of every file under d."""
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*"))
            if p.is_file()}


# ================================================================================================= with Blender
@needs_blender
def test_a_built_case_is_a_project_with_blenders_frames_and_its_bake(tmp_path, capsys):
    code, out = cli(["parity", "lab", "view_standard", "--root", str(tmp_path), "--build", "--refs-only"], capsys)
    assert code == 0
    d = tmp_path / "view_standard"
    assert out["built"]["view_standard"]["frames"] == {"main": 1} and out["built"]["view_standard"]["bake"]["version"] >= 4
    for rel in ("mk.toml", "case.json", "scene.blend", "scene.bake/bake.json", "ref/main/00001.png",
                "ref16/main/00001.png", "linear/main/00001.exr"):
        assert (d / rel).is_file(), rel
    bake = json.loads((d / "scene.bake" / "bake.json").read_text())
    assert bake["cut"]["main"][0]["camera"] == "lab@main" and bake["frames"] == 1
    rec = json.loads((d / "case.json").read_text())
    assert rec["report"]["eevee"]["taa_render_samples"] == 64

    ref16, ref8 = PY.read(d / "ref16" / "main" / "00001.png"), PY.read(d / "ref" / "main" / "00001.png")
    assert ref16.shape == (540, 960, 3) and ref8.dtype == np.uint8
    p = json.loads((d / "case.json").read_text())["case"]["params"]
    x = (np.arange(960) + 0.5) / 960
    linear = 2.0 ** (p["lo"] + (p["hi"] - p["lo"]) * x)
    white = ref16[540 // 16, :, 0] / 255.0                              # the middle of the first strip (white)
    seen = linear < 1.0                                                 # Standard clips at 1
    assert np.abs(white[seen] - srgb(linear[seen])).max() * 255 < 1.0  # the film's 1.5 px filter barely bends a ramp
    m = PY.metrics(PY.stats(ref16, ref8))
    assert m["max"] <= 2.0 and abs(np.mean(m["bias"])) < 0.6           # the 8-bit frame: dither and rounding only


@needs_blender
def test_a_variant_renders_the_saved_scene_again_and_leaves_the_case_alone(tmp_path, capsys):
    code, _ = cli(["parity", "lab", "contact_shadow", "--root", str(tmp_path), "--build", "--refs-only"], capsys)
    assert code == 0
    d = tmp_path / "contact_shadow"
    before = {k: digest(d / k) for k in ("ref", "ref16", "linear", "scene.bake")}
    blend = hashlib.sha256((d / "scene.blend").read_bytes()).hexdigest()
    for name, extra in (("noshadow", []), ("cycles", ["--variant-samples", "16", "--seed", "3"])):
        code, out = cli(["parity", "lab", "contact_shadow", "--root", str(tmp_path), "--variant", name, *extra], capsys)
        assert code == 0 and out["rendered"]["contact_shadow"]["frames"] == {"main": 2}, out
        v = d / "variants" / name
        for rel in ("variant.json", "main/00001.png", "main/00002.png", "main/00001.exr", "16/main/00002.png"):
            assert (v / rel).is_file(), f"{name}/{rel}"
        assert PY.read(v / "16" / "main" / "00001.png").dtype == np.float32
    assert {k: digest(d / k) for k in before} == before                 # the case's frames and bake untouched
    assert hashlib.sha256((d / "scene.blend").read_bytes()).hexdigest() == blend

    def luma(name, f=1):
        return PY.read(d / name / "main" / f"{f:05d}.png").astype(np.float32) @ PY.LUMA
    term = luma("ref") - luma("variants/noshadow")                      # the shadow term: <= 0, and deep somewhere
    assert (term > 2).mean() < 0.002 and (term < -40).mean() > 0.01, (term < -40).mean()
    rep = json.loads((d / "variants" / "cycles" / "variant.json").read_text())["report"]
    cy = rep["cycles"]
    assert rep["engine"] == "CYCLES" and cy["samples"] == 16 and cy["max_bounces"] == 0 and not cy["use_denoising"]
    assert not cy["use_adaptive_sampling"] and cy["filter_width"] == pytest.approx(1.5) and cy["seed"] == 3
    # direct light only: where EEVEE's shadow is black and its light is not, Cycles' is black too (no bounce fills it)
    umbra = (luma("ref") < 1) & (luma("variants/noshadow") > 50)
    assert umbra.sum() > 1000 and float(np.median(luma("variants/cycles")[umbra])) < 3.0


# ================================================================================================= with a stand-in mkr
FAKE = '''#!{python}
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd")
r = sub.add_parser("render")
for k in ("--bake", "--output", "--frames", "--out", "--samples", "--percent", "--shutter", "--params"):
    r.add_argument(k)
r.add_argument("--motion-blur", action="store_true")
sub.add_parser("params")
a = p.parse_args()
params = json.loads(Path(a.params).read_text()) if getattr(a, "params", None) else {{}}
with open(os.environ["FAKE_MKR_LOG"], "a") as fh:
    fh.write(json.dumps({{"argv": sys.argv[1:], "params": params}}) + "\\n")
if a.cmd == "params":
    print(json.dumps({{"exposure": 0.0, "shadows": True}}))
    sys.exit(0)
case = Path(a.bake).parent
src = (case / "ref" if params.get("shadows", True) else case / "variants" / "noshadow") / a.output
off = 40.0 * (params.get("exposure", 0.3) - 0.3)
frames = []
for part in a.frames.split(","):
    lo, _, hi = part.partition(":")
    frames += list(range(int(lo), int(hi or lo) + 1))
for f in frames:
    im = np.asarray(Image.open(src / f"{{f:05d}}.png").convert("RGB")).astype(np.float32)
    Image.fromarray(np.clip(np.rint(im + off), 0, 255).astype(np.uint8)).save(Path(a.out) / f"{{f:05d}}.png")
print(json.dumps({{"rendered": len(frames), "seconds": 0.01, "s_per_frame": 0.01, "gpu_ms_per_frame": 0.0,
                  "adapter": "fake", "warnings": []}}))
'''


@pytest.fixture
def mkr(tmp_path, monkeypatch):
    path = tmp_path / "fake_mkr"
    path.write_text(FAKE.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "mkr_calls.jsonl"
    monkeypatch.setenv("MK_MKR", str(path))
    monkeypatch.setenv("FAKE_MKR_LOG", str(log))
    return lambda: [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


H, W = 16, 24
SHADE = (slice(4, 12), slice(8, 16))                                 # where the shadow falls: 64 of 384 pixels


def _frame(lit, shadow=None):
    im = np.full((H, W, 3), lit, np.uint8)
    if shadow is not None:
        im[SHADE] = shadow
    return im


def _case(lab, name, frames):
    """A built case as the lab keeps one (frames 1 and 2 of two are Blender's), without Blender; `frames` {folder:
    image} for both frames, a variant's folder with its variant.json."""
    d = lab / name
    case = {"name": name, "kind": "light", "doc": "", "frames": 2, "refs": [1, 2], "labels": ["high", "grazing"],
            "outputs": [{"name": "main", "size": [W, H]}], "samples": 64}
    (d / "scene.bake").mkdir(parents=True)
    (d / "scene.blend").write_bytes(b"")
    (d / "case.json").write_text(json.dumps({"case": case, "report": {"bake": {"version": 5}}}))
    for folder, im in frames.items():
        for f in (1, 2):
            png(d / folder / "main" / f"{f:05d}.png", im)
        if folder.startswith("variants/"):
            (d / folder / "variant.json").write_text(json.dumps({"variant": {"name": folder.split("/")[1]}}))
    return d


def _write(path, doc):
    path.write_text(json.dumps(doc))
    return path


def test_against_measures_mkr_with_a_variant_instead_of_eevees_frames(tmp_path, capsys, mkr):
    lab = tmp_path / "lab"
    _case(lab, "area_square", {"ref": _frame(100), "variants/cycles": _frame(106)})
    _case(lab, "area_disk", {"ref": _frame(100)})
    code, out = cli(["parity", "lab", "area_square", "area_disk", "--root", str(lab), "--against", "cycles"], capsys)
    assert code == 0 and out["against"] == "cycles"
    m = out["cases"]["area_square"]["main"]["overall"]
    assert m["mean_abs"] == pytest.approx(6.0) and m["bias"] == pytest.approx([-6.0] * 3)   # mkr (EEVEE's copy) - Cycles
    assert out["notes"]["area_disk"] == "no cycles variant: `mk parity lab --variant cycles area_disk`"
    summary = lab / "area_square" / "variants" / "cycles" / "compare" / "main" / "summary.json"
    assert json.loads(summary.read_text())["ref"].endswith("variants/cycles/main")
    assert not (lab / "area_square" / "compare").exists()            # the comparison with EEVEE's frames stays as it was
    assert [c["argv"][c["argv"].index("--bake") + 1] for c in mkr()] == [str(lab / "area_square" / "scene.bake")]


def test_separate_splits_the_error_into_light_and_shadow(tmp_path, capsys, mkr):
    lab = tmp_path / "lab"
    # Cycles: lit 120, a black umbra 2 (term -118); EEVEE: its light 2 levels too bright, 12 levels of light in the
    # umbra (term -108); mkr: EEVEE's frames 10 levels brighter (exposure 0.55): its light 12 off Cycles', its term EEVEE's.
    _case(lab, "area_square", {"variants/cycles_noshadow": _frame(120), "variants/cycles": _frame(120, 2),
                               "variants/noshadow": _frame(122), "ref": _frame(122, 14)})
    probes = _write(tmp_path / "probes.json", {"area_square": {
        "lines": [{"name": "left edge", "output": "main", "frame": 1, "y": 8, "x": [2, 14], "rows": 3},
                  {"name": "top edge", "output": "main", "frame": 1, "x": 12, "y": [0, 10], "rows": 3}],
        "boxes": [{"name": "in the shadow", "output": "main", "frame": 2, "box": [9, 5, 15, 11]}]}})
    code, out = cli(["parity", "lab", "area_square", "--root", str(lab), "--separate", "--per-frame", "--params",
                     str(_write(tmp_path / "p.json", {"exposure": 0.55})), "--probes", str(probes)], capsys)
    assert code == 0, out
    params = [c["params"] for c in mkr() if c["argv"][0] == "render"]
    assert params == [{"exposure": 0.55}, {"exposure": 0.55, "shadows": False}]   # as it is, then shadows off
    c = out["cases"]["area_square"]
    mean = {k: v["main"]["overall"]["mean_abs"] for k, v in c["comparisons"].items()}
    assert mean == pytest.approx({"eevee_cycles": (64 * 12 + 320 * 2) / 384, "mkr_cycles": (64 * 22 + 320 * 12) / 384,
                                  "mkr_eevee": 10, "eevee_cycles_ns": 2, "mkr_cycles_ns": 12, "mkr_eevee_ns": 10})
    assert c["comparisons"]["eevee_cycles_ns"]["main"]["overall"]["bias"] == pytest.approx([2.0] * 3)  # EEVEE - Cycles
    sh = c["shadow"]["outputs"]["main"]["all"]
    assert sh["pixels"] == 128 and sh["share"] == pytest.approx(100 * 64 / 384, abs=1e-3)
    assert {k: v[0] for k, v in sh["depth"].items()} == pytest.approx({"eevee": -108, "cycles": -118, "mkr": -108})
    assert {k: (v["mean_abs"], v["bias"][0]) for k, v in sh["pairs"].items()} == {
        "eevee_cycles": (10.0, 10.0), "mkr_cycles": (10.0, 10.0), "mkr_eevee": (0.0, 0.0)}
    assert sh["leak"] == pytest.approx({"eevee": 12.0, "mkr": 22.0})   # light in Cycles' umbra
    assert sh["acne"] == {"eevee": {"mean": 0.0, "over": 0.0}, "mkr": {"mean": 0.0, "over": 0.0}}
    (line, top), box = c["probes"]["lines"], c["probes"]["boxes"][0]
    assert line["measured"]["cycles"] == {"width": 0.8, "dark": 2.0, "bright": 120.0}    # a sharp edge: 0.8 px
    assert top["measured"]["mkr"] == {"width": 0.8, "dark": 24.0, "bright": 132.0}       # down a column
    assert set(line["measured"]) == {"eevee", "cycles", "mkr"}                          # shadowed frames only
    assert box["luma"] == pytest.approx({"eevee": 14, "eevee_ns": 122, "cycles": 2, "cycles_ns": 120, "mkr": 24,
                                         "mkr_ns": 132}, abs=0.01)
    assert box["darkening"] == pytest.approx({"eevee": 108, "cycles": 118, "mkr": 108}, abs=0.01)
    rows = {r["frame"]: r for r in out["rows"]}
    assert set(rows) == {None, 1, 2} and rows[2]["label"] == "grazing"
    assert rows[1]["comparisons"]["mkr_eevee"]["mean_abs"] == 10 and rows[1]["shadow"]["pixels"] == 64
    head, _, whole, *per_frame = LB.separation_table(out["rows"], per_frame=True).splitlines()
    assert head.split()[3:9] == ["E-C", "M-C", "M-E", "En-Cn", "Mn-Cn", "Mn-En"] and len(per_frame) == 2
    assert whole.split()[3:15] == ["3.67", "+3.67", "13.67", "+13.67", "10.00", "+10.00", "2.00", "+2.00", "12.00",
                                   "+12.00", "10.00", "+10.00"]
    assert whole.split("|")[1].split() == ["16.67", "-108.0", "-118.0", "-108.0", "10.0", "10.0", "0.0", "+12.0", "+22.0"]
    assert (lab / "area_square" / "mkr_noshadow" / "main" / "00002.png").is_file()
    assert (lab / "area_square" / "separate" / "mkr_cycles" / "main" / "sheet_01.png").is_file()


def test_separate_without_the_variants_says_which_are_missing(tmp_path, capsys, mkr):
    lab = tmp_path / "lab"
    _case(lab, "area_square", {"ref": _frame(122, 52), "variants/noshadow": _frame(122)})
    code, out = cli(["parity", "lab", "area_square", "--root", str(lab), "--separate", "--no-sheet"], capsys)
    assert code == 0
    c = out["cases"]["area_square"]
    assert set(c["comparisons"]) == {"mkr_eevee", "mkr_eevee_ns"}
    assert c["missing"]["eevee_cycles"] == ["variants/cycles"]
    sh = c["shadow"]["outputs"]["main"]["all"]
    assert c["shadow"]["renderers"] == ["eevee", "mkr"] and sh["leak"] == {} and sh["acne"] == {}   # no truth to lean on
    assert sh["pairs"]["mkr_eevee"]["mean_abs"] == 0 and set(sh["pairs"]) == {"mkr_eevee"}
    assert out["notes"]["area_square"] == ("no cycles, cycles_noshadow variant: "
                                          "`mk parity lab --variant NAME area_square`")


def test_the_labs_probes_lie_inside_their_cases_frames():
    for name, probes in LB.PROBES.items():
        case = LB.spec(name)
        for p in probes.get("lines", []):
            n = p["rows"]
            box = ((p["x"][0], p["y"] - n // 2, p["x"][1], p["y"] - n // 2 + n) if isinstance(p["x"], list)
                   else (p["x"] - n // 2, p["y"][0], p["x"] - n // 2 + n, p["y"][1]))
            assert LB._inside(case, p, box), (name, p["name"])
        for p in probes.get("boxes", []):
            assert LB._inside(case, p, p["box"]), (name, p["name"])
    with pytest.raises(LB.LabError):
        LB._inside(LB.spec("penumbra"), {"name": "off", "output": "main", "frame": 1}, (900, 10, 980, 20))
    with pytest.raises(LB.LabError):
        LB._inside(LB.spec("penumbra"), {"name": "no frame 3", "output": "main", "frame": 3}, (0, 0, 10, 10))


def test_the_dof_cases_key_every_frame_and_the_probe_lies_in_its_picture():
    for name in ("dof", "dof_psf"):
        case = LB.spec(name)
        keys = case["params"]["keys"]
        assert len(keys) == case["frames"] == len(case["labels"])
        for fstop, blades, rotation, ratio, focus in keys:              # f-stop, blades, rotation, ratio, focus
            assert fstop > 0 and (blades == 0 or blades >= 3) and ratio > 0 and focus > 0
    for card in LB.spec("dof_psf")["params"]["cards"]:                  # seen from (0, -2, 1) along +Y, 85 mm, 960 x 540
        x, y, z = card["at"]
        u, v = 480 + x * LB.DOF_PX / (y + 2.0), 270 - (z - 1.0) * LB.DOF_PX / (y + 2.0)
        assert 0 < u < 960 and 0 < v < 540, card["name"]


def test_the_fast_gi_cases_trace_as_mks_scenes_and_key_a_view_or_a_roughness_per_frame():
    for name in ("gi_wall", "gi_cabin", "gi_rough"):
        case = LB.spec(name)
        assert case["eevee"] == LB.RT and case["world"]["props"]["sun_threshold"] == 0.0     # mk's worlds take no sun
        assert len(case["labels"]) == case["frames"] and "lights" not in case["params"]      # the world lights alone
        views = case["params"]["camera"]
        assert len(views) == case["frames"] if isinstance(views[0][0], list) else len(views) == 2
    floor = LB.spec("gi_rough")["params"]["objects"][0]["material"]
    assert floor["keys"]["Roughness"] == LB.GI_ROUGH and len(LB.GI_ROUGH) == LB.spec("gi_rough")["frames"]


def test_the_lit_fast_gi_cases_light_an_l_and_thin_strands_and_each_variant_changes_one_setting():
    for name in ("gi_seat", "gi_strands"):
        case = LB.spec(name)
        assert case["eevee"] == LB.RT and case["world"]["color"] == [0.1] * 3                  # a dim world, one light
        assert len(case["params"]["lights"]) == 1
        assert len(case["labels"]) == case["frames"] == len(case["params"]["camera"])
    strands = [o for o in LB.spec("gi_strands")["params"]["objects"] if o["kind"] == "tube"]
    assert {o["r"] for o in strands} == {0.003} and len(strands) == 28
    for v, key, value in (("fastgi_res1", "fast_gi_resolution", "1"), ("fastgi_res4", "fast_gi_resolution", "4"),
                          ("fastgi_thin", "fast_gi_thickness_near", 0.02), ("fastgi_far", "fast_gi_thickness_far", 0.1)):
        eevee = LB.VARIANTS[v]["eevee"]
        assert eevee[key] == value and {k: x for k, x in eevee.items() if k != key} == {
            k: x for k, x in LB.RT.items() if k != key}, v
    assert LB.VARIANTS["fastgi_ao_res1"]["eevee"]["fast_gi_method"] == "AMBIENT_OCCLUSION_ONLY"
    assert not LB.VARIANTS["fastgi_noshadow"]["shadows"] and not LB.VARIANTS["fastgi_ao_noshadow"]["shadows"]


def test_the_clamp_case_keys_one_floor_a_frame_under_strips_past_the_clamp():
    case = LB.spec("rt_clamp")
    assert case["eevee"] == LB.RT and LB.VARIANTS["noclamp"]["eevee"] == {"clamp_surface_indirect": 0.0}
    floor, strips = case["params"]["objects"][0], case["params"]["objects"][1:]
    assert all(len(v) == case["frames"] == len(case["labels"]) for v in floor["material"]["keys"].values())
    strengths = sorted({s["material"]["inputs"]["Strength"] for s in strips})
    assert strengths[0] < 10.0 < strengths[-1]          # under and over Blender's clamp_surface_indirect (10)


def test_the_emitter_case_lights_one_card_width_a_frame_under_the_clamp():
    case = LB.spec("rt_emit")
    assert case["eevee"] == LB.RT and len(case["labels"]) == case["frames"] == len(LB.RT_EMIT_FRAMES)
    cards = [o for o in case["params"]["objects"] if o["kind"] == "box"]
    for f, (w, r) in enumerate(LB.RT_EMIT_FRAMES):
        on = [c for c in cards if c["material"]["keys"]["Strength"][f] > 0.0]
        assert len(on) == 2 and all(c["size"][0] == w for c in on)          # that width over each floor
        assert all(0.0 < c["material"]["keys"]["Strength"][f] < 10.0 for c in on)   # under clamp_surface_indirect
    for floor in case["params"]["objects"][:2]:
        assert floor["material"]["keys"]["Roughness"] == [r for _, r in LB.RT_EMIT_FRAMES]
    assert LB.VARIANTS["trace_thick"]["eevee"]["ray_tracing_options"]["use_denoise"] is False


def test_a_penumbra_is_measured_10_to_90_percent_from_the_dark_side():
    ramp = np.concatenate([np.full(5, 20.0), np.linspace(20, 220, 11), np.full(5, 220.0)])   # 10 samples wide
    assert LB.step_width(ramp) == {"width": 8.0, "dark": 20.0, "bright": 220.0}
    assert LB.step_width(ramp[::-1])["width"] == pytest.approx(8.0)     # from either side
    assert LB.step_width(np.full(7, 3.0)) is None


def test_lab_variant_flags_are_checked(tmp_path, capsys):
    root = str(tmp_path / "lab")
    for argv, says in ((["--variant", "eevee_hd"], "no such variant"), (["--against", "nope"], "no such variant"),
                       (["--variant", "cycles", "--separate"], "--variant only renders"),
                       (["--against", "cycles", "--separate"], "one at a time"),
                       (["--seed", "2"], "go with --variant"), (["--probes", "p.json"], "goes with --separate")):
        code, out = cli(["parity", "lab", "area_square", "--root", root, *argv], capsys)
        assert code == 2 and says in out["error"], (argv, out)
