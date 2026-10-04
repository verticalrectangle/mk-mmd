"""mk grip: solve how a character's hand holds a prop (pen, wheel, pinch, rest, neck) and write the result as JSON."""
import argparse
import json
import time
from pathlib import Path

from .. import bridge
from .. import config as CFG
from ..cache import Cache, key as cache_key
from ..core import jsonx
from ..project import ProjectError
from .common import CHECK_FAILED, add_project_arg, emit, get_project, scene_path, UsageError

STYLES = ("pen", "wheel", "pinch", "rest", "neck")
PROP_FLAGS = {"pen": ("length", "radius", "tip", "nib_offset"), "wheel": ("radius", "tube"), "pinch": ("width",),
              "rest": ("surface",), "neck": ()}                 # prop keys with a flag of their own, per style
SOLVER_FLAGS = {"pen": ("posture",), "wheel": ("approach", "wrap"), "pinch": ("edge",), "rest": ("face",), "neck": ()}
EXPORT_VERSION = 1          # bump with the `hand_model` op's output

HELP = """Solve a grip: finger rotations and the hand/prop frame for a character's hand on a prop.

The hand's rest skin and bones are exported from the scene (Blender op hand_model, cached), then solved with numpy and
scipy on the model's own skin until the fingers touch the prop and nothing penetrates. The result is JSON:

  bones             {blender bone: [w, x, y, z]}  pose-bone rotation_quaternion for every finger joint bone (bone-local,
                    relative to rest: key them as they are, the wrist is not driven)
  target_in_wrist   4x4 grip frame in the wrist bone's rest frame at the wrist head. In a build, the wrist's world
                    matrix is  grip_frame_world @ inverse(target_in_wrist)
  report            contacts (gap_mm per finger/region), penetration_mm, finger_clash_mm, angles_deg, style numbers
  frame_world_quat  (pen with a writing posture) the grip frame's world rotation while writing

Styles and the grip frame (prop keys: a prop card's use.grip entry; unknown keys are ignored):
  pen    prop length, radius (m or a [[distance from nib, radius], ...] profile), tip, nib_offset [x, y, z]. Frame: the
         prop's own: +Z nib -> cap, +X the barrel side facing the back of the hand. Lateral tripod; with --posture
         (nib, shoulder, pole, table, ...) the hand's writing orientation is solved too.
  wheel  prop radius (ring), tube. Torus power grip: frame on the tube's centreline at the palm, x radial outward,
         y tangent, z ring axis. --approach DEG (palm side in the section plane: 0 = +x, 90 = +z), --wrap +1/-1.
  pinch  prop width (thickness between the pads). Thumb-index pad pinch: frame midway between the pads, z from the index
         pad to the thumb pad, x away from the wrist. --edge M (pads' distance inside the object's edge).
  rest   prop surface = plane. Relaxed hand lying on a plane: frame on the plane below the palm, z = the plane's
         normal, x = the hand's heading. --face palm|back.
  neck   a fretting hand on a guitar neck: --card (a prop card, or its use.grip entry of type neck, as JSON or @file) with
         --chord (power, E, A, D, G, C ... or a table) and --fret (the position fret under the index finger) make the prop;
         or --prop with the solver's own neck problem. Frame: on the board under the position fret's wire, x toward the nut,
         z out of the board, y = z cross x. The thumb behind the neck, pressing fingers arched on their strings.

A project's mk.toml names the armature (--cast); results are cached by a hash of the hand and the inputs. Exit codes:
0 ok, 1 the grip misses a gate (a contact gap, penetration or finger clash over its limit; the file is still written),
2 usage error, 3 Blender or runtime error.

Examples:
  mk grip pen build/scene.blend --cast reisen --side R --length 0.14 --radius 0.0068 --out tracks/grip_pen.json
  mk grip wheel --cast driver --side L --radius 0.19 --tube 0.015 --approach 90 --out grips/wheel_L.json
  mk grip pinch --armature Model_arm --side R --width 0.011 --out grips/strap.json
  mk grip rest --side L --prop '{"surface": "plane"}' --face palm --out grips/rest_L.json
"""


def add(sub):
    p = sub.add_parser("grip", help="solve a hand grip on a prop (pen, wheel, pinch, rest, neck)", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("style", choices=STYLES)
    p.add_argument("scene", nargs="?", metavar="SCENE.blend", help="default: the project's scene")
    who = p.add_argument_group("whose hand")
    who.add_argument("--cast", metavar="NAME", help="cast member of the project (default: its only member)")
    who.add_argument("--armature", metavar="NAME", help="armature object (default: the scene's only MMD armature)")
    who.add_argument("--side", choices=("L", "R", "l", "r"), default="R", help="which hand (default R)")
    who.add_argument("--skin-radius", type=float, default=0.16, metavar="M",
                     help="skin vertices within this of the wrist head are solved against (default 0.16)")
    who.add_argument("--frame", type=int, help="Blender frame at which the armature's world matrix is read")
    prop = p.add_argument_group("the prop (flags override keys of --prop)")
    prop.add_argument("--prop", metavar="JSON|@FILE", help="prop geometry as JSON, or @file.json")
    prop.add_argument("--length", type=float, metavar="M", help="pen: nib to cap")
    prop.add_argument("--radius", type=float, metavar="M", help="pen: barrel radius; wheel: ring radius")
    prop.add_argument("--tip", type=float, metavar="M", help="pen: length of the conical tip")
    prop.add_argument("--nib-offset", type=float, nargs=3, metavar=("X", "Y", "Z"), help="pen: nib in the prop frame")
    prop.add_argument("--tube", type=float, metavar="M", help="wheel: tube radius")
    prop.add_argument("--width", type=float, metavar="M", help="pinch: thickness between the pads")
    prop.add_argument("--surface", help="rest: the surface kind (plane)")
    prop.add_argument("--card", metavar="JSON|@FILE", help="neck: a prop card (or its neck grip entry)")
    prop.add_argument("--chord", help="neck: a chord name (power, E, A, D, G, C, Em, Am ...) or a table as JSON")
    prop.add_argument("--fret", type=int, help="neck: the position fret (under the index finger)")
    how = p.add_argument_group("how to solve")
    how.add_argument("--params", metavar="JSON|@FILE", help="solver keyword arguments as JSON, or @file.json")
    how.add_argument("--posture", metavar="JSON|@FILE", help="pen: writing posture (nib, shoulder, pole, table, ...)")
    how.add_argument("--approach", type=float, metavar="DEG", help="wheel: direction from the tube to the palm")
    how.add_argument("--wrap", type=int, choices=(-1, 1), help="wheel: the way the fingers wrap round the tube")
    how.add_argument("--edge", type=float, metavar="M", help="pinch: pads' distance inside the object's edge")
    how.add_argument("--face", choices=("palm", "back"), help="rest: which side of the hand lies on the surface")
    how.add_argument("--seeds", type=int, help="parallel starts (the best wins)")
    how.add_argument("--workers", type=int, help="processes for the seeds (default: all cores; 1 = in this process)")
    gate = p.add_argument_group("gates (exit 1 when missed)")
    gate.add_argument("--max-gap", type=float, default=3.0, metavar="MM", help="contact gaps (default 3)")
    gate.add_argument("--max-penetration", type=float, default=1.0, metavar="MM", help="default 1")
    gate.add_argument("--max-clash", type=float, default=1.0, metavar="MM", help="finger inside finger, default 1")
    p.add_argument("--out", metavar="FILE.json", help="write the result here (else it is printed)")
    p.add_argument("--no-cache", action="store_true", help="solve again even when the same inputs were solved before")
    add_project_arg(p)
    p.set_defaults(func=run)


def _json_arg(text, what):
    if text is None:
        return {}
    try:
        raw = Path(text[1:]).expanduser().read_text(encoding="utf-8") if text.startswith("@") else text
        val = json.loads(raw)
    except (OSError, json.JSONDecodeError) as e:
        raise UsageError(f"{what} is not valid JSON ({e})")
    if not isinstance(val, dict):
        raise UsageError(f"{what} must be a JSON object")
    return val


def _own(args, style, what):
    """The flags of `style` (from `what`: style -> flags) that were given; a flag only other styles have is a usage
    error."""
    for owner, flags in what.items():
        for flag in flags:
            if flag not in what[style] and getattr(args, flag) is not None:
                raise UsageError(f"--{flag.replace('_', '-')} is an option of the {owner} style, not {style}")
    return [f for f in what[style] if getattr(args, f) is not None]


def _neck_prop(args):
    """The solver's neck problem from --card, --chord and --fret."""
    from ..core import fretting as FR
    card = _json_arg(args.card, "--card")
    entry = card
    if "use" in card:                                            # a whole prop card: its neck grip entry
        entry = next((g for g in card["use"].get("grip", []) if g.get("type") == "neck"), None)
        if entry is None:
            raise UsageError("--card has no use.grip entry of type neck")
    if not (args.chord and args.fret):
        raise UsageError("neck needs --chord and --fret (or --prop with the solver's own neck problem)")
    try:
        chord = json.loads(args.chord) if args.chord.lstrip().startswith("{") else args.chord
        return FR.solver_prop(entry, chord, args.fret)[0]
    except (FR.FrettingError, KeyError, ValueError) as e:
        raise UsageError(f"neck: {e}")


def _prop(args):
    if args.style != "neck" and any(getattr(args, k, None) is not None for k in ("card", "chord", "fret")):
        raise UsageError("--card, --chord and --fret are options of the neck style")
    if args.style == "neck" and getattr(args, "card", None):
        return _neck_prop(args)
    prop = _json_arg(args.prop, "--prop")
    for flag in _own(args, args.style, PROP_FLAGS):
        value = getattr(args, flag)
        prop[flag] = list(value) if flag == "nib_offset" else value
    return prop


def _params(args):
    params = _json_arg(args.params, "--params")
    for flag in _own(args, args.style, SOLVER_FLAGS):
        params[flag] = _json_arg(args.posture, "--posture") if flag == "posture" else getattr(args, flag)
    for flag in ("seeds", "workers"):
        if getattr(args, flag) is not None:
            params[flag] = getattr(args, flag)
    return params


def _armature(args, proj):
    """The armature name to export (None: the scene's only MMD armature)."""
    if args.armature:
        return args.armature
    if args.cast or (proj is not None and proj.cast):
        if proj is None:
            raise UsageError("--cast needs a project (mk.toml); pass --armature instead")
        try:
            return proj.cast_member(args.cast).get("armature") or None
        except ProjectError as e:
            raise UsageError(str(e))
    return None


def problems(result, max_gap, max_penetration, max_clash):
    """What a grip misses: contact gaps, penetration and finger clash over their limits (mm)."""
    rep, out = result["report"], []
    if rep.get("penetration_mm", 0.0) > max_penetration:
        out.append(f"penetration {rep['penetration_mm']} mm > {max_penetration}")
    if rep.get("finger_clash_mm", 0.0) > max_clash:
        out.append(f"finger clash {rep['finger_clash_mm']} mm > {max_clash}")
    for name, c in (rep.get("contacts") or {}).items():
        gap = c.get("gap_mm") if isinstance(c, dict) else None
        if gap is not None and gap > max_gap:
            out.append(f"contact {name}: gap {gap} mm > {max_gap}")
    return out


def run(args):
    from ..solvers import grip as G
    t0 = time.time()
    proj = get_project(args)
    blend = scene_path(args.scene, proj)
    side = args.side.upper()
    armature = _armature(args, proj)
    prop, params = _prop(args), _params(args)
    cache = Cache(proj.mk_dir / "cache" if proj else CFG.cache_dir() / "cache")
    model_key = cache_key("hand_model", EXPORT_VERSION, blend, armature, side, args.skin_radius, args.frame)
    npz = cache.path("hand_model", model_key, "npz")
    if not npz.exists():
        tmp = npz.with_name(npz.stem + ".part.npz")
        bridge.run("hand_model", {"armature": armature, "side": side, "radius": args.skin_radius,
                                  "frame": args.frame, "out": str(tmp)}, blend=blend, project=proj, timeout=600)
        tmp.replace(npz)
    hand = G.HandModel.from_npz(npz)
    solve_key = cache_key("grip", G.VERSION, args.style, hand.digest(), prop,
                          {k: v for k, v in params.items() if k != "workers"})
    result = None if args.no_cache else cache.load_json("grip", solve_key)
    cached = result is not None
    if not cached:
        try:
            result = G.solve(args.style, hand, prop, **params)
        except (ValueError, TypeError) as e:
            raise UsageError(f"{args.style}: {e}")
        result = json.loads(jsonx.dumps(result, precision=7))
        cache.save_json("grip", solve_key, result)
    bad = problems(result, args.max_gap, args.max_penetration, args.max_clash)
    out = {"ok": not bad, "style": args.style, "side": side, "armature": armature, "scene": str(blend),
           "seconds": round(time.time() - t0, 2), "cached": cached, "bones": len(result["bones"]),
           "target_in_wrist": result["target_in_wrist"], "report": result["report"]}
    if "frame_world_quat" in result:
        out["frame_world_quat"] = result["frame_world_quat"]
    if bad:
        out["problems"] = bad
    if args.out:
        path = Path(args.out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        out["out"] = str(path.resolve())
    else:
        out["result"] = result
    emit(out)
    return 0 if not bad else CHECK_FAILED
