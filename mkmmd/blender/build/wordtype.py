"""wordtype: sung words as type (docs/design.md: Text, `lyrics`). Called by the text stage for a `[[text]]` entry with
`lyrics = {...}`; returns one ordinary text spec per word, built by `mkmmd.core.wordtype` from the timeline.

    [[text]]
    name = "mug"
    mount = "mug"; at = [0, 0, 0.14]; facing = [0.5, -0.8, 0.2]; box = [0.16, 0.2]      # placement and look: as any text
    font = "fraunces_italic"; fit = 0.8; color = "love"; glow = 0.8
    lyrics = {timeline = "audio/timeline.json", line = 1, words = [1, 4], style = "rise", from = 1.3, to = 2.6,
              layout = {kind = "stack", dir = "up", rows = 4}, colors = ["love", "gold", "pine"]}

The timeline is read here and nowhere else; the words never reach a log line, an error, a report or an object property
(objects are `<name>_l<line>w<word>`, `..._<shot>` with `lyrics.zones`, and carry `mk_lyric = [line, word]`). With `zones` the
cut (the project's `[[shot]]` names and times) is handed to the expansion, and every zone's panel is asked for on its own.
A ring laid round a point of the scene (`layout = {kind = "ring", center = <target>}`) gets an empty, `<name>_ring`, keyed on
every frame its words are up where that point is: in the world for type of the scene, seen through the cut's camera of
the output for screen type; the words' trees read it (mkmmd/blender/build/textring.py)."""
import json

from ...core import screentype as SR
from ...core import wordtype as LY
from . import BuildError, collection


def _timeline(ctx, spec):
    name = spec.get("name", "?")
    ref = (spec.get("lyrics") or {}).get("timeline")
    if not ref:
        raise BuildError(f"text {name!r}: lyrics needs `timeline` = the path of the timeline JSON")
    path = ctx.path(ref)
    cache = ctx.__dict__.setdefault("_lyric_timelines", {})
    if path not in cache:
        try:
            with open(path, encoding="utf-8") as fh:
                cache[path] = json.load(fh)
        except OSError as e:
            raise BuildError(f"text {name!r}: cannot read the timeline {path}: {e.strerror}") from None
        except ValueError as e:                     # JSONDecodeError: its message holds a position, not the content
            raise BuildError(f"text {name!r}: the timeline {path} is not valid JSON ({type(e).__name__})") from None
    return cache[path]


def _panel(ctx, spec):
    """(w, h) metres of the entry's surface or box, checking every surface of an `on` list on the way."""
    from . import text as TX
    probe = {k: v for k, v in spec.items() if k != "lyrics"}
    on = spec.get("on")
    refs = on if isinstance(on, (list, tuple)) else [on]
    panels = []
    for ref in refs:
        if ref is None:
            probe.pop("on", None)
        else:
            probe["on"] = ref
        panels.append(TX._place(ctx, probe)[3])
    return panels[0]


def _shots(ctx):
    """The project's cut as [{name, from, to}] clip seconds, for `lyrics.zones` (a shot that only lends frames has none)."""
    return [{"name": s["name"], "from": s["from"], "to": s["to"]} for s in ctx.data.get("shot", [])
            if s.get("name") and "from" in s and "to" in s]


def expand(ctx, spec, measure=None, report=None):
    """The word texts of a lyrics entry. `measure` (string[, font] -> ink box, see mkmmd.core.wordtype.expand) serves the
    layouts that put words next to each other; `report` (a dict) receives the summary: counts and (line, word) numbers.
    With `lyrics.bullets` the words fly past a target, read in the scene on the frame each word lands."""
    name = spec.get("name", "?")
    tl = _timeline(ctx, spec)
    lyr = spec.get("lyrics") or {}
    own = lyr.get("zones") is not None or bool(lyr.get("bullets"))      # zones: each zone's panel; bullets: their box
    try:
        specs, summary = LY.expand(spec, tl, fps=ctx.fps, frame0=ctx.frame0, panel=None if own else _panel(ctx, spec),
                                   measure=measure, shots=_shots(ctx), panel_of=lambda e: _panel(ctx, e),
                                   marks=lambda at, times: _marks(ctx, at, times, name))
    except LY.LyricsError as e:
        raise BuildError(str(e)) from None
    for key in ("first_frame", "last_frame", "cut_frame"):      # clip frames -> Blender frames (what `mk look` takes)
        if summary.get(key) is not None:
            summary[key] += ctx.frame0
    for b in summary.get("blocks", []) + summary.get("rings", []):
        for key in ("first_frame", "last_frame"):
            if b.get(key) is not None:
                b[key] += ctx.frame0
    if summary.get("ring_track") is not None:
        _ring_track(ctx, spec, specs, summary)
    if report is not None:
        report.update(summary)
    ctx.log("lyrics", name, f"{summary['words']} word(s) of line(s) {summary['lines']}, Blender frames "
            f"{summary['first_frame']}..{summary['last_frame']}, cut at {summary['cut_frame']}"
            + (f", {len(summary['blocks'])} shot(s)" if summary.get("blocks") else "")
            + (f", {len(summary['rings'])} ring(s)" if summary.get("rings") else "")
            + (", flying past their target" if summary.get("bullets") else ""))
    return specs


def _ring_track(ctx, spec, specs, summary):
    """A ring round a point of the scene: the empty `<name>_ring` (`@<output>` for one output of several) keyed on every
    frame from the first word's landing to the cut where the target is, the world point for type of the scene, its place in
    the picture through the cut's camera of the output for screen type (frame heights from the centre, the screen plane's
    units); every word's `orbit` tracks it. A point behind the camera keeps the last place it had."""
    import bpy
    from .. import keys as K
    name, target = spec["name"], summary["ring_track"]["target"]
    out = spec.get("_aspect")
    ename = f"{name}_ring" + (f"@{out}" if out is not None and spec.get("_multi") else "")
    if bpy.data.objects.get(ename) is not None:
        raise BuildError(f"text {name!r}: an object named {ename!r} exists already (the empty the ring turns about)")
    f0, f1 = summary["first_frame"], summary["cut_frame"]
    if not specs or f0 is None or f1 <= f0:
        return
    screen = SR.screen_spec(spec) is not None
    seen = _ring_points(ctx, name, target, f0, f1)
    pts = seen["world"] if not screen else seen["screen"].get(out)
    if pts is None:
        raise BuildError(f"text {name!r}: a screen ring round a point of the scene needs the cut's cameras for output "
                         f"{out!r} (a [[shot]])")
    ob = bpy.data.objects.new(ename, None)
    collection("Text").objects.link(ob)
    ob.empty_display_size = 0.05
    frames = list(range(f0, f1 + 1))
    for i in range(3):
        K.key_prop(ob, "location", frames, [float(p[i]) for p in pts], index=i, interp="LINEAR")
    for s in specs:
        s["kinetic"]["orbit"]["track"] = ename
    summary["ring_track"] = {"object": ename, "frames": [f0, f1]}


def _ring_points(ctx, name, target, f0, f1):
    """{"world": [(x, y, z)], "screen": {output: [(u, v, 0)]}} of `target` on Blender frames f0..f1: the world point, and its
    place through the camera the cut has on each frame for every output (frame heights from the picture's centre, x right,
    y up), worked out once for all the outputs of an entry and kept on the build."""
    import bpy
    from .. import fisheye as FE
    from .. import scene as S
    from . import targets
    cache = ctx.__dict__.setdefault("_ring_points", {})
    key = (json.dumps(target, sort_keys=True), f0, f1)
    if key in cache:
        return cache[key]
    sc = bpy.context.scene
    cut = sorted((e for e in S.shot_table(sc) if not e.get("plate")), key=lambda e: e["from"])
    sizes = {o["name"]: tuple(o["size"]) for o in ctx.project.get("outputs") or []}
    outs = sorted({o for e in cut for o in e["cameras"]})
    world, screen = [], {o: [] for o in outs}
    keep = sc.frame_current
    try:
        for f in range(f0, f1 + 1):
            sc.frame_set(f)
            p = targets.point(ctx, target)
            world.append(tuple(p))
            shot = next((e for e in reversed(cut) if e["from"] <= f), cut[0] if cut else None)
            for o in outs:
                cam = bpy.data.objects.get(shot["cameras"].get(o) or "")
                size = sizes.get(o, (sc.render.resolution_x, sc.render.resolution_y))
                v = FE.view(sc, cam, p, size) if cam is not None else None
                screen[o].append(None if v is None or v.z <= 0 else ((v.x - 0.5) * size[0] / size[1], v.y - 0.5, 0.0))
    except BuildError as e:
        raise BuildError(f"text {name!r}: ring center: {e}") from None
    finally:
        sc.frame_set(keep)
    for o, pts in screen.items():               # behind the camera: where it was last (or is first) seen
        seen = [p for p in pts if p is not None] or [(0.0, 0.0, 0.0)]
        last = seen[0]
        for i, p in enumerate(pts):
            last = pts[i] = p if p is not None else last
    cache[key] = {"world": world, "screen": screen}
    return cache[key]


def _marks(ctx, target, times, name):
    """The world point (x, y, z) of `target` at each clip second of `times`, the scene set to the frame of each."""
    import bpy
    from . import targets
    sc = bpy.context.scene
    keep = sc.frame_current
    out = []
    try:
        for t in times:
            sc.frame_set(int(round(ctx.frame(t))))
            out.append(tuple(targets.point(ctx, target)))
    except BuildError as e:
        raise BuildError(f"text {name!r}: lyrics bullets at: {e}") from None
    finally:
        sc.frame_set(keep)
    return out
