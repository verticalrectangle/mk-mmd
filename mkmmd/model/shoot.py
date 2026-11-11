"""Pictures of a built model in the renderer its videos are drawn with (bpy-free): what `mk model shoot` does.

A model is judged where it will be seen. `project` writes a small mk project round a model's PMX: a grey paper sweep, a
key, a fill and a rim light, a shot per view, and the model turned on its root between the shots (she turns, the lights
and the camera stay, as on a turntable: every view is lit alike). The lights are as strong as shows a texture's colours
as they are painted under the Standard view transform (`view`; AgX, mk's default for a video, pales what is bright
and saturated). `shoot` writes it, runs `mk build` and `mk look` (mkr draws one frame per view) and lays the pictures
out in sheets.

A view frames a region of the model (`FRAMES`: the whole figure, the bust, the face, the legs, the feet) from an angle
(`VIEWS`: the model's turn in degrees, positive toward her left, and the camera's height above the region's middle in
degrees). The regions are placed by the model's own measures (its rig.json `measure`, or `measure_pmx`): the floor, the
top of everything it has (ears and horns included), its eye height and its head bone, so a small model fills its
pictures as a tall one does. `expressions` adds the face from the front once per morph (the model's own morph names)."""
import json
import math
import subprocess
import sys
from pathlib import Path

FPS = 30
HOLD = 6                                  # frames a view lasts; its picture is the frame in its middle
SAMPLES = 64                              # mk look's samples: the area lights' soft shadows without grain
LENS = {"full": 85.0, "bust": 85.0, "face": 100.0, "legs": 85.0, "feet": 85.0}
SENSOR = 36.0                             # mm across a square frame
# name -> (region, the model's turn, the camera's elevation), degrees
VIEWS = {
    "front": ("full", 0.0, 0.0), "3q": ("full", -35.0, 3.0), "side": ("full", -90.0, 0.0), "back": ("full", 180.0, 0.0),
    "back_3q": ("full", 145.0, 4.0), "right_3q": ("full", 35.0, 3.0),
    "bust": ("bust", 0.0, 0.0), "bust_3q": ("bust", -35.0, 3.0), "bust_back": ("bust", 150.0, 3.0),
    "face": ("face", 0.0, 0.0), "face_3q": ("face", -35.0, 0.0), "face_side": ("face", -90.0, 0.0),
    "face_right": ("face", 35.0, 0.0), "face_low": ("face", -25.0, -18.0), "face_high": ("face", -30.0, 28.0),
    "face_back": ("face", 160.0, 6.0),
    "legs": ("legs", -30.0, 4.0), "feet": ("feet", -30.0, 22.0),
}
SETS = {
    "turn": ["front", "3q", "side", "back", "back_3q"],
    "head": ["face", "face_3q", "face_side", "face_low", "face_high", "face_back"],
    "all": ["front", "3q", "side", "back", "back_3q", "bust", "bust_3q", "bust_back", "face", "face_3q", "face_side",
            "face_low", "face_high", "face_back", "legs", "feet"],
}
# name, colour, watts, where, size, casts shadows. The last two stand in for the ambient light a scene has: without
# them what faces away from the three lamps (the inside of the hair seen from behind) would be black
LIGHTS = (("key", "#fff6ee", 120, (1.5, -2.8, 2.6), 1.8, True), ("fill", "#eceeff", 46, (-2.2, -2.4, 1.5), 2.2, True),
          ("rim", "#ffffff", 64, (-1.2, 2.6, 2.4), 1.2, True),
          ("ambient_front", "#f4f2f6", 32, (0.6, -3.0, 0.3), 4.0, False),
          ("ambient_back", "#f4f2f6", 27, (1.6, 3.0, 1.4), 4.0, False))


class ShootError(Exception):
    pass


def views_of(spec):
    """The view names a comma list asks for: names of VIEWS and of SETS (`turn`, `head`, `all`), in order, once each."""
    out = []
    for word in [w.strip() for w in (spec or "all").split(",") if w.strip()]:
        names = SETS.get(word, [word])
        for n in names:
            if n not in VIEWS:
                raise ShootError(f"view {n!r}: one of {', '.join(VIEWS)}, or a set: {', '.join(SETS)}")
            if n not in out:
                out.append(n)
    return out


def measure_pmx(path, unit=0.08):
    """{top, width, eye_height, head_z, hip_height} of a .pmx in metres (`unit` metres per PMX unit), for a model without
    a rig.json: its vertices' extent, its eye bones (両目, else 頭 + a fifth of the head above it), 頭 and 下半身."""
    from . import pmx_io
    m = pmx_io.read(str(path))
    zs = [v.position[1] * unit for v in m.vertices]
    xs = [v.position[0] * unit for v in m.vertices]
    bone = {b.name: b for b in m.bones}

    def z(name):
        return bone[name].position[1] * unit if name in bone else None
    top, head = max(zs), z("頭")
    if head is None:
        raise ShootError(f"{path}: no 頭 bone to frame the face by")
    eye = z("両目")
    if eye is None or not head < eye < top:
        eye = head + 0.2 * (top - head)
    hip = z("下半身")
    return {"top": top, "width": max(xs) - min(xs), "eye_height": eye, "head_z": head,
            "hip_height": hip if hip is not None else 0.5 * top}


def regions(measure):
    """{region: (the height its middle is at, the height of the field it fills)} in metres from a model's measures. The
    head's own size is taken from the eyes' height over the head bone (hair, ears and horns do not count)."""
    top, eye, head = float(measure["top"]), float(measure["eye_height"]), float(measure["head_z"])
    hip = float(measure.get("hip_height", 0.5 * top))
    hu = 4.4 * (eye - head)                                 # about chin to crown
    return {"full": (0.5 * top, max(1.08 * top, 1.04 * float(measure.get("width", 0.0)))),
            "bust": (eye - 0.55 * hu, 3.4 * hu),
            "face": (eye + 0.10 * hu, 1.9 * hu),
            "legs": (0.5 * hip + 0.02, 1.2 * hip),
            "feet": (0.07, 0.42)}


def shots(measure, views, expressions=()):
    """[{name, at, look, lens, yaw, morph}]: a camera per view, in front of the model (-y), and the model's turn; then the
    face from the front once per morph of `expressions`."""
    reg = regions(measure)
    out = []
    for name in list(views) + [f"morph:{m}" for m in expressions]:
        region, yaw, elev = VIEWS["face"] if name.startswith("morph:") else VIEWS[name]
        mid, field = reg[region]
        lens = LENS[region]
        dist = field * lens / SENSOR
        e = math.radians(elev)
        out.append({"name": name, "at": [0.0, round(-dist * math.cos(e), 4), round(mid + dist * math.sin(e), 4)],
                    "look": [0.0, 0.0, round(mid, 4)], "lens": lens, "yaw": yaw,
                    "morph": name[6:] if name.startswith("morph:") else None})
    return out


def frame_of(i):
    """The Blender frame view `i`'s picture is taken on."""
    return 1 + i * HOLD + HOLD // 2


def _s(text):
    return json.dumps(str(text), ensure_ascii=False)


def project(pmx, rig, plan, size=1440, color="#cfd2da", view="Standard"):
    """The mk.toml of the look-dev project: `plan` is `shots()`'s list, `pmx` and `rig` the model's files (rig None for
    a model without a rig.json), `size` the square pictures' side in pixels at 100 %, `view` the view transform."""
    dt = HOLD / FPS
    out = ["# Written by mk model shoot: a model on a turntable, a shot per view (mkmmd/model/shoot.py).", "[project]",
           'name = "shoot"', f"fps = {FPS}", "frame0 = 1", f"duration = {len(plan) * dt:.4f}",
           'blend = "build/shoot.blend"', "", "[[output]]", 'name = "sq"', f"size = [{int(size)}, {int(size)}]", "",
           "[look]", 'palette = "rose-pine-dawn"', f"view = {_s(view)}", "", "[scene]", "start = 1", "", "[[set]]",
           'name = "paper"', 'kind = "sweep"', "width = 14", "depth = 9", "height = 6", "radius = 2.0",
           f"color = {_s(color)}", "", "[[cast]]", 'name = "her"', f"pmx = {_s(pmx)}"]
    if rig:
        out.append(f"rig = {_s(rig)}")
    out.append("")
    for name, colour, power, at, sz, shadow in LIGHTS:
        out += ["[[light]]", f'name = "{name}"', 'kind = "area"', f'color = "{colour}"', f"power = {power}",
                f"at = {list(at)}", "look = [0.0, 0.0, 1.1]", f"size = {sz}", f"shadow = {str(shadow).lower()}", ""]
    for i, s in enumerate(plan):
        out += ["[[shot]]", f'name = "v{i:02d}"', f"from = {i * dt:.4f}", f"to = {(i + 1) * dt:.4f}", f"at = {s['at']}",
                f"look = {s['look']}", f"lens = {s['lens']}", ""]
    keys = ", ".join(f"[{i * dt:.4f}, {math.radians(s['yaw']):.5f}]" for i, s in enumerate(plan))
    out += ["[[key]]", 'target = "Her"', 'prop = "rotation_euler"', "index = 2", 'interp = "CONSTANT"',
            f"keys = [{keys}]", ""]
    faces = [(i, s["morph"]) for i, s in enumerate(plan) if s["morph"]]
    if faces:
        ex = []
        for i, morph in faces:
            a, b = i * dt, (i + 1) * dt
            keyed = ([[round(a - 0.001, 4), 0.0]] if i else []) + [[round(a, 4), 1.0], [round(b - 0.002, 4), 1.0],
                                                                 [round(b - 0.001, 4), 0.0]]
            ex.append(f"{{ morph = {_s(morph)}, keys = {keyed} }}")
        out += ["[perform.her]", "blink = { per_min = 0.01 }           # none while the pictures are taken",
                "expressions = [" + ", ".join(ex) + "]", ""]
    return "\n".join(out)


def sheet(cells, out, cols, cell=480, label=True):
    """A contact sheet of [(label, picture path)], `cols` across, written to `out`."""
    from PIL import Image, ImageDraw

    from .. import fonts
    rows = (len(cells) + cols - 1) // cols
    im = Image.new("RGB", (cols * cell, rows * cell), (32, 32, 36))
    draw, font = ImageDraw.Draw(im), fonts.label_font(max(cell // 30, 12))
    for i, (name, path) in enumerate(cells):
        with Image.open(path) as pic:
            pic = pic.convert("RGB").resize((cell, cell), Image.LANCZOS)
        x, y = (i % cols) * cell, (i // cols) * cell
        im.paste(pic, (x, y))
        if label:
            draw.text((x + 8, y + 6), name, fill=(40, 40, 48), font=font)
    im.save(out, quality=92)
    return out


def run_mk(args, cwd, log):
    """Run `mk ARGS` in `cwd`, its output into the file `log`; a ShootError with the log's end when it fails."""
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.run([sys.executable, "-m", "mkmmd.cli.main", *args], cwd=cwd, stdout=f,
                              stderr=subprocess.STDOUT).returncode
    if code:
        tail = Path(log).read_text(encoding="utf-8", errors="replace")[-1500:]
        raise ShootError(f"mk {args[0]} failed (exit {code}; {log}):\n{tail}")


def shoot(pmx, out, views=None, expressions=(), size=1080, rig=None, unit=0.08, color="#cfd2da", view="Standard"):
    """The pictures of a model: `out`/<view>.jpg for each view (and `morph_<name>.jpg` per expression), the sheets
    `out`/sheet.jpg (every picture) and, when there are that many, sheet_turn.jpg and sheet_head.jpg. Returns the
    report {views: {name: path}, sheets, measure, project}."""
    pmx, out = Path(pmx).expanduser().resolve(), Path(out).expanduser().resolve()
    if not pmx.is_file():
        raise ShootError(f"{pmx}: no such model (build it first: mk model build)")
    if rig is None:
        beside = pmx.with_suffix(".rig.json")
        rig = beside if beside.is_file() else None
    measure = json.loads(Path(rig).read_text(encoding="utf-8")).get("measure") if rig else None
    if not measure or "eye_height" not in measure or "head_z" not in measure:
        measure = measure_pmx(pmx, unit)
    names = views_of(views)
    plan = shots(measure, names, expressions)
    proj = out / "project"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "mk.toml").write_text(project(pmx, str(rig) if rig else None, plan, size=size, color=color, view=view),
                                  encoding="utf-8")
    run_mk(["build"], proj, proj / "build.log")
    pics = proj / "pictures"
    frames = [frame_of(i) for i in range(len(plan))]
    run_mk(["look", "--frames", ",".join(str(f) for f in frames), "--size", str(int(size)), "--samples", str(SAMPLES),
            "--out", str(pics)], proj, proj / "look.log")
    got = {}
    for s, f in zip(plan, frames):
        src = pics / f"cut_sq_{f:05d}.jpg"
        if not src.is_file():
            raise ShootError(f"view {s['name']}: mk look drew no {src.name} (see {proj / 'look.log'})")
        dst = out / (s["name"].replace("morph:", "morph_") + ".jpg")
        dst.write_bytes(src.read_bytes())
        got[s["name"]] = str(dst)
    sheets = {"all": str(sheet(list(got.items()), out / "sheet.jpg", cols=min(len(got), 5 if len(got) > 8 else 4)))}
    for key in ("turn", "head"):
        part = [(n, got[n]) for n in SETS[key] if n in got]
        if len(part) >= 3 and len(part) < len(got):
            sheets[key] = str(sheet(part, out / f"sheet_{key}.jpg", cols=len(part), cell=640))
    faces = [(n[6:], p) for n, p in got.items() if n.startswith("morph:")]
    if faces:
        sheets["expressions"] = str(sheet(faces, out / "sheet_expressions.jpg", cols=min(len(faces), 6), cell=400))
    return {"views": got, "sheets": sheets, "measure": {k: round(float(v), 4) for k, v in measure.items()},
            "project": str(proj)}
