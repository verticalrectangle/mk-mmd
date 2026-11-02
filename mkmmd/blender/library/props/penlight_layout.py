"""Layout of the idol fan penlight (pure Python + numpy, no bpy: importable from tests): its meshes, the colour roles, the
light's colours and the card.

Frame (metres): the penlight stands on the end of its handle along +Z, the end at z = 0 and the tip of the glowing tube
at z = LENGTH; its front (-Y) is the side with the button. A fan penlight of the common kind in proportion: 265 mm long,
a 31 mm handle a fist closes round (soft ribs on the grip, a pill button above the fist where the thumb reaches it, a
small ring at the end for a wrist strap), a clear collar, and a 24.4 mm frosted tube with a domed tip that glows in the
colour picked. Every part is a lathe or a rounded panel. No logo or lettering.

Colour roles (`slots` of the prop: a palette slot, a hex or an `a:b:t` mix): handle (base), grip (overlay: the soft
ribbed band), button (text), collar (hl_high), ring (subtle). The light: `colors`, a list of colours (default love, gold,
foam, iris, rose, pine, text), one of which lights the tube: the root property `color` is an index into it (rounded,
wrapping round the list: 7 is 0 again with seven colours), keyed CONSTANT so the light switches on a hit as a real
penlight's button does; `glow` (0 .. 1) is its level (0: the tube is frosted plastic, unlit).

Card: `use.grip` `hold` (type "hold": the span of the handle a fist closes on, `axis` +Z toward the tube), `use.anchor`
`tip` (the tube's tip, `dir` +Z: where a light trail starts) and `light` (the middle of the tube), `use.look` `tube` and
`grip`; `params` (color, glow), `colors` (the light's colours as given)."""
import numpy as np

from ....core import shell as S
from . import showkit as SK

LENGTH = 0.265
HANDLE_R = 0.0155                    # the grip's radius (a 31 mm handle)
HANDLE_TOP = 0.104                   # where the handle meets the collar
GRIP = (0.016, 0.080)                # the soft ribbed band
RIBS = 6                             # its ribs
RIB_D = 0.0007                       # how deep the grooves between them go
HOLD = (0.0125, 0.0875)              # the span a fist closes on: the end cap to just under the button
BUTTON_Z = 0.0935                    # the button's centre, on the front (-Y)
BUTTON = (0.0075, 0.013, 0.0022)     # its width (x), height (z) and how far it stands out
COLLAR = (HANDLE_TOP, 0.112, 0.0168)  # the clear collar: from, to, radius
TUBE_R = 0.0122                      # the glowing tube's radius
TUBE_FROM = 0.109                    # it starts inside the collar
RING_R, RING_T = 0.0062, 0.0011      # the strap ring under the end cap: radius and wire
STRENGTH = 2.0                       # the tube's emission at glow 1 (option `strength`): its colour still reads
ROLES = {"handle": "base", "grip": "overlay", "button": "text", "collar": "hl_high", "ring": "subtle"}
COLORS = ["love", "gold", "foam", "iris", "rose", "pine", "text"]
MATS = tuple(ROLES) + ("tube",)      # material index -> role (the tube's material is the light)
PARAMS = {"color": (0.0, "", "the light's colour: an index into `colors` (rounded, wrapping round the list); key it "
                             "CONSTANT to switch on a hit"),
          "glow": (1.0, "", "the light's level, 0 .. 1 (0: off, frosted plastic)")}


def options(slots):
    """(colors [spec, ...], params {color, glow}, strength) from the `slots` of the [[prop]], checked: a bad value is a
    ValueError that says what to change."""
    slots = slots or {}
    colors = slots.get("colors", COLORS)
    if isinstance(colors, str):
        colors = [colors]
    if not isinstance(colors, (list, tuple)) or not 1 <= len(colors) <= 16:
        raise ValueError(f"penlight: colors is a list of 1 .. 16 colours (palette slots, '#hex' or 'a:b:t') (got {colors!r})")
    for i, c in enumerate(colors):
        SK.rgb(c, slots, f"penlight: colors[{i}]")
    P = {"color": SK.number(slots, "color", PARAMS["color"][0], "penlight"),
         "glow": SK.number(slots, "glow", PARAMS["glow"][0], "penlight", 0.0, 10.0)}
    return list(colors), P, SK.number(slots, "strength", STRENGTH, "penlight", 0.0, 1000.0)


def pick(color, n):
    """The index of the lit colour for the root property `color` among n colours: rounded, wrapping round the list (what
    the tube's material computes)."""
    return int(np.floor(float(color) + 0.5)) % int(n)


def handle_profile():
    """(r, z) of the handle, axis to axis: a gently domed end cap rolling into the side, the ribbed grip (shallow round
    grooves between full rims), a shoulder up to the collar."""
    z0, z1 = GRIP
    head = [(0.0, 0.0), (0.0100, 0.0006), (0.0145, 0.0030), (HANDLE_R, 0.0080), (HANDLE_R, z0)]
    t = np.linspace(0.0, 1.0, RIBS * 8 + 1)[1:-1]
    ribs = [(HANDLE_R - 0.5 * RIB_D * (1.0 - np.cos(2.0 * np.pi * RIBS * u)), z0 + (z1 - z0) * u) for u in t]
    tail = [(HANDLE_R, z1), (HANDLE_R, HANDLE_TOP - 0.004), (0.0152, HANDLE_TOP), (0.0, HANDLE_TOP)]
    P = np.array(head + ribs + tail)
    r = np.zeros(len(P))
    r[1:4] = (0.004, 0.003, 0.003)
    r[-3] = 0.002                                 # the shoulder's roll
    return S.fillet(P, r, 3)


def handle_mesh(seg=48):
    """The handle (material handle) with its ribbed band in the grip's material."""
    m = S.lathe(handle_profile(), seg=seg, mat=MATS.index("handle"))
    m.assign(MATS.index("grip"), lambda c, n: (c[:, 2] > GRIP[0] - 1e-4) & (c[:, 2] < GRIP[1] + 1e-4))
    return m


def button_mesh():
    """The pill button on the front of the handle: a domed rounded panel, its back sunk into the handle."""
    w, h, t = BUTTON
    pill = S.rounded_panel(S.stadium(w, h, 6), t + 0.0015, r_edge=0.0009, n=3, dome=0.0004, mat=MATS.index("button"))
    # the panel lies in x-y with its face toward +Z: stand it up on the front of the handle, its face toward -Y
    V = pill.V
    out = S.Mesh(np.stack([V[:, 0], -(HANDLE_R - 0.0015) - V[:, 2], BUTTON_Z + V[:, 1]], 1), pill.Q, pill.T, pill.Qm,
                 pill.Tm)
    return out if out.volume() > 0 else out.flip()


def collar_profile():
    z0, z1, r = COLLAR
    return S.fillet(np.array([(0.0, z0), (r, z0), (r, z1), (0.0, z1)]), [0.0, 0.0012, 0.0012, 0.0], 3)


def collar_mesh(seg=48):
    return S.lathe(collar_profile(), seg=seg, mat=MATS.index("collar"))


def tube_profile(n_dome=10):
    """(r, z) of the glowing tube: a straight frosted tube from inside the collar up to a domed tip at LENGTH."""
    a = np.linspace(0.0, 0.5 * np.pi, n_dome + 1)
    dome = [(TUBE_R * np.cos(t), LENGTH - TUBE_R + TUBE_R * np.sin(t)) for t in a]
    return np.array([(0.0, TUBE_FROM), (TUBE_R, TUBE_FROM)] + dome[:-1] + [(0.0, LENGTH)])


def tube_mesh(seg=48):
    return S.lathe(tube_profile(), seg=seg, mat=MATS.index("tube"))


def ring_mesh():
    """The strap ring under the end cap: a wire loop in the x-z plane, its top through the cap's rim."""
    a = np.linspace(0.0, 2.0 * np.pi, 33)[:-1]
    path = np.stack([RING_R * np.cos(a), np.zeros_like(a), -RING_R + 0.0015 + RING_R * np.sin(a)], 1)
    m = S.tube(path, RING_T, sides=8, closed=True, caps=(False, False), mat=MATS.index("ring"), up=(0.0, 1.0, 0.0))
    m = m.weld(1e-9)                              # the closed sweep repeats its first section at the end
    return m if m.volume() > 0 else m.flip()


def meshes():
    """{part: Mesh} of the penlight's objects: body (handle, button, collar, ring) and tube (the light)."""
    return {"body": S.merge([handle_mesh(), button_mesh(), collar_mesh(), ring_mesh()]), "tube": tube_mesh()}


def card(name, slots=None, colors=None, params=None):
    """The prop card (docs/design.md: Prop card): the hold, the tip and the light, the look points."""
    hold = {"name": "hold", "type": "hold", "center": [0.0, 0.0, round(sum(HOLD) / 2, 5)], "axis": [0.0, 0.0, 1.0],
            "length": round(HOLD[1] - HOLD[0], 5), "radius": HANDLE_R}
    mid = round(0.5 * (COLLAR[1] + LENGTH), 5)
    return {"size": [round(2 * COLLAR[2], 4), round(2 * COLLAR[2], 4), LENGTH], "origin": "tail", "front": "-Y",
            "slots": dict(slots or {}),
            "use": {"grip": [hold],
                    "anchor": [{"name": "tip", "point": [0.0, 0.0, LENGTH], "dir": [0.0, 0.0, 1.0]},
                               {"name": "light", "point": [0.0, 0.0, mid], "dir": [0.0, 0.0, 1.0]}],
                    "look": [{"name": "tube", "point": [0.0, 0.0, mid]},
                             {"name": "grip", "point": [0.0, 0.0, round(sum(HOLD) / 2, 5)]}]},
            "params": dict(params or {k: v[0] for k, v in PARAMS.items()}), "colors": list(colors or COLORS),
            "objects": [f"{name}_body", f"{name}_tube"]}
