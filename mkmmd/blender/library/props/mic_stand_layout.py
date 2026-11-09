"""Layout of the boom microphone stand with its microphone (pure Python + numpy, no bpy: importable from tests): the
numbers, every mesh, the cord's path, the colour roles and the card.

Frame (metres): the stand's round base lies on the floor, its middle at the ORIGIN, Z up. The pole stands on it; at its
top a swivel carries the boom, which rises toward +Y at `angle` degrees above level (its short tail with the counterweight
toward -Y). A clip at the boom's front end holds a handheld vocal microphone (the `handheld_mic`'s profiles) in line with
the boom, its ball grille toward the singer. So the singer stands on the +Y side, facing -Y: turn the stand with `yaw`.

Options (`slots`): `height` (m, 1.30: the height of the grille's centre, where a mouth goes), `angle` (degrees, 28) and
`boom` (m, 0.36: the boom from the swivel to the clip); the pole's length follows from them. The card's look point
`grille` is the grille's centre (put the stand so that it is 4 to 6 cm in front of her lips) and `reach` its level
distance from the pole.

Colour roles (a palette slot or a hex): stand (subtle: the tubes, satin metal), base (base: the cast base, the swivel,
the clutch, the clip: dark), body, grille, hardware and cable (the microphone's and its cord's, as `handheld_mic`)."""
import math

import numpy as np

from ....core import shell as S
from . import handheld_mic_layout as MIC

ROLES = {"stand": "subtle", "base": "base", "body": "base", "grille": "text", "hardware": "subtle", "cable": "text"}
MATS = {name: i for i, name in enumerate(ROLES)}
OPTIONS = {"height": 1.30, "angle": 28.0, "boom": 0.36}

BASE_R, BASE_H = 0.125, 0.022           # the cast base: a low dome
POLE_R = (0.0110, 0.0080)               # the lower tube and the upper one that slides in it
CLUTCH = {"z": 0.62, "r": 0.0165, "h": 0.045}       # the height clutch between the tubes
SWIVEL = {"r": 0.0150, "w": 0.030}      # the boom's swivel: a drum across the pole's top, its axis along x
BOOM_R = 0.0070
BOOM_TAIL = 0.14                        # the boom behind the swivel
WEIGHT = {"r": 0.0150, "l": 0.055}      # the counterweight on the boom's tail
CLIP = {"r_in": MIC.HANDLE[0] + 0.0035, "t": 0.0035, "l": 0.034, "stem": 0.014}     # the clip's sleeve round the handle, on a
#                                                                                     # stem that stands this far off the boom
CABLE_R = MIC.CABLE_R


def options(slots=None):
    """`OPTIONS` with a prop's `slots` laid over them, checked."""
    slots = slots or {}
    out = {k: float(slots.get(k, v)) for k, v in OPTIONS.items()}
    if not 0.6 <= out["height"] <= 2.2:
        raise ValueError(f"mic_stand: height = {out['height']:g}: the grille's height in metres, 0.6 to 2.2")
    if not -10.0 <= out["angle"] <= 75.0:
        raise ValueError(f"mic_stand: angle = {out['angle']:g}: degrees the boom rises, -10 to 75")
    if not 0.1 <= out["boom"] <= 0.8:
        raise ValueError(f"mic_stand: boom = {out['boom']:g}: the boom's length in metres, 0.1 to 0.8")
    return out


def frame(opt=None):
    """Where things are for the options: {d (the boom's unit direction), up (square to it, in the y-z plane), pivot (the
    swivel's middle, the pole's top), end (where the clip's stem stands on the boom), off (the mic's axis above the
    boom's), grip (the middle of the clip, on the mic's axis), tail (the mic's tail: its frame's origin), grille (the
    grille's centre), reach}. The mic lies over the boom's end, parallel to it."""
    o = options(opt)
    a = math.radians(o["angle"])
    d = np.array([0.0, math.cos(a), math.sin(a)])
    up = np.array([0.0, -math.sin(a), math.cos(a)])
    held = MIC.GRIP_Z                                                   # the clip closes round the handle here
    off = BOOM_R + CLIP["stem"] + CLIP["r_in"] + CLIP["t"]
    rel = (o["boom"] + MIC.BALL_C - held) * d + off * up                # the swivel to the grille's centre
    grille = np.array([0.0, rel[1], o["height"]])
    pivot = grille - rel
    if pivot[2] < CLUTCH["z"] + CLUTCH["h"] + 0.05:
        raise ValueError(f"mic_stand: height {o['height']:g} with angle {o['angle']:g} and boom {o['boom']:g} leaves the "
                         f"pole's top at {pivot[2]:.2f} m, under its clutch: raise it or lower the boom")
    end = pivot + o["boom"] * d
    grip = end + off * up
    return {"d": d, "up": up, "pivot": pivot, "end": end, "off": off, "grip": grip, "tail": grip - held * d,
            "grille": grille, "reach": float(grille[1])}


def _lathe(pts, seg, mat, r=0.0008):
    return S.lathe(S.fillet(np.asarray(pts, float), r, 3), seg=seg, mat=mat)


def meshes(opt=None):
    """{object suffix: core.shell.Mesh} in the prop's frame: base, pole (both tubes and the clutch), boom (the swivel, the
    tube, the counterweight, the clip), mic_body, mic_grille, mic_connector."""
    f = frame(opt)
    d, up, pivot, end = f["d"], f["up"], f["pivot"], f["end"]
    m = MATS
    base = _lathe([(0.0, 0.0), (BASE_R, 0.0), (BASE_R, 0.006), (BASE_R - 0.012, 0.014), (0.045, BASE_H - 0.002),
                   (0.020, BASE_H), (0.0, BASE_H)], 64, m["base"], r=0.003)
    zt = float(pivot[2])
    lower = S.tube(np.array([[0.0, 0.0, BASE_H - 0.003], [0.0, 0.0, CLUTCH["z"]]]), POLE_R[0], sides=20, up=(1.0, 0.0, 0.0),
                   mat=m["stand"])
    upper = S.tube(np.array([[0.0, 0.0, CLUTCH["z"]], [0.0, 0.0, zt]]), POLE_R[1], sides=16, up=(1.0, 0.0, 0.0), mat=m["stand"])
    c = CLUTCH
    clutch = _lathe([(0.0, 0.0), (POLE_R[0] + 0.001, 0.0), (c["r"], 0.006), (c["r"], c["h"] - 0.010),
                     (POLE_R[1] + 0.0015, c["h"]), (0.0, c["h"])], 24, m["base"], r=0.002).moved((0.0, 0.0, c["z"] - 0.012))
    s = SWIVEL
    drum = _lathe([(0.0, 0.0), (s["r"] - 0.002, 0.0), (s["r"], 0.003), (s["r"], s["w"] - 0.003), (s["r"] - 0.002, s["w"]),
                   (0.0, s["w"])], 24, m["base"], r=0.0012)
    drum = S.place(drum, pivot - np.array([0.5 * s["w"], 0.0, 0.0]), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    knob = _lathe([(0.0, 0.0), (0.006, 0.0), (0.011, 0.004), (0.011, 0.014), (0.007, 0.018), (0.0, 0.018)], 16, m["base"],
                  r=0.0015)
    knob = S.place(knob, pivot + np.array([0.5 * s["w"], 0.0, 0.0]), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    tube = S.tube(np.stack([pivot - BOOM_TAIL * d, end + 0.014 * d]), BOOM_R, sides=14, up=(1.0, 0.0, 0.0), mat=m["stand"])
    w = WEIGHT
    weight = _lathe([(0.0, 0.0), (w["r"] - 0.003, 0.0), (w["r"], 0.004), (w["r"], w["l"] - 0.004), (w["r"] - 0.003, w["l"]),
                     (0.0, w["l"])], 20, m["base"], r=0.002)
    weight = S.place(weight, pivot - (BOOM_TAIL + 0.012) * d, d, (1.0, 0.0, 0.0))
    k = CLIP
    ro = k["r_in"] + k["t"]
    top = BOOM_R + k["stem"] + 0.003                                      # the stem: from the boom's axis into the sleeve
    stem = _lathe([(0.0, 0.0), (0.0085, 0.0), (0.0085, BOOM_R + 0.003), (0.0050, BOOM_R + 0.007), (0.0050, top),
                   (0.0, top)], 16, m["base"], r=0.001)
    stem = S.place(stem, end, up, (1.0, 0.0, 0.0))
    sleeve = _lathe([(0.0, 0.0), (ro - 0.001, 0.0), (ro, 0.002), (ro, k["l"] - 0.002), (ro - 0.001, k["l"]),
                     (0.0, k["l"])], 28, m["base"], r=0.0006)             # the sleeve: a solid drum, the handle fills it
    sleeve = S.place(sleeve, f["grip"] - 0.5 * k["l"] * d, d, (1.0, 0.0, 0.0))
    boom = S.merge([drum, knob, tube, weight, stem, sleeve])
    put = lambda mesh: S.place(mesh, f["tail"], d, (1.0, 0.0, 0.0))        # noqa: E731  the mic's +Z along the boom
    return {"base": base, "pole": S.merge([lower, upper, clutch]), "boom": boom,
            "mic_body": put(_lathe(MIC.body_profile(), 48, m["body"])),
            "mic_grille": put(_lathe(MIC.grille_profile(), 48, m["grille"], r=0.0004)),
            "mic_connector": put(_lathe(MIC.connector_profile(), 32, m["hardware"]))}


def cable_path(opt=None):
    """(n, 3) points of the cord: out of the mic's tail, back under the boom to the swivel, down the pole and away along
    the floor toward -Y."""
    f = frame(opt)
    d, up, pivot, tail = f["d"], f["up"], f["pivot"], f["tail"]
    under = -(BOOM_R + CABLE_R + 0.002) * up
    pts = [tail - 0.002 * d, tail - 0.040 * d - 0.006 * up, tail - 0.075 * d - (f["off"] + 0.014) * up,
           pivot + 0.5 * (f["end"] - pivot) + under - 0.02 * up, pivot + 0.05 * d + under,
           pivot + np.array([0.0, -0.03, -0.05])]
    x = POLE_R[0] + CABLE_R + 0.004
    pts += [np.array([0.0, -x, z]) for z in (pivot[2] - 0.20, 0.5 * (pivot[2] + CLUTCH["z"]), CLUTCH["z"] - 0.10, 0.25)]
    pts += [np.array([0.0, -(BASE_R - 0.03), BASE_H + 0.02]), np.array([0.0, -(BASE_R + 0.05), CABLE_R]),
            np.array([0.03, -(BASE_R + 0.30), CABLE_R]), np.array([-0.04, -(BASE_R + 0.60), CABLE_R])]
    return np.array(pts, float)


def bounds(opt=None):
    P = np.concatenate([m.V for m in meshes(opt).values()] + [cable_path(opt)])
    return P.min(0), P.max(0)


def card(name, slots=None, opt=None):
    """The prop card (docs/design.md: Prop card): the look points `grille` (the grille's centre: where a mouth goes) and
    `pole` (the swivel), `reach` (the grille's level distance from the pole), the size."""
    f = frame(opt)
    lo, hi = bounds(opt)
    r = lambda v: [round(float(x), 5) for x in v]                          # noqa: E731
    return {"size": r(hi - lo), "origin": "floor_center", "front": "+Y", "slots": dict(slots or {}),
            "reach": round(f["reach"], 5),
            "use": {"look": [{"name": "grille", "point": r(f["grille"])}, {"name": "pole", "point": r(f["pivot"])}]}}
