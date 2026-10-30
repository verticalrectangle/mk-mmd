"""Ring type and wobbling letters in a kinetic word's node tree (docs/design.md: Text, Lyrics: rings and wobble; the numbers
are mkmmd/core/typering.py and mkmmd/core/typewobble.py). Called by the text stage (build/text.py) while it builds the tree
of a word whose `kinetic` table has `bend`, `orbit` or `wobble`; `g` is that stage's node helper (`node`, `put`, `math`).

  bend    the filled letters (and the strip behind them) follow a circle through the word's pivot: a point x along the word
          and y up from its middle goes to the angle x / R round a circle of radius R + y about a centre R below the pivot
          (`side` 1, the tops out) or R - y about a centre above it (`side` -1, the tops in)
  orbit   after the word's own keyed transform, a turn about the ring's centre by the keyed `Orbit` degrees; with `track` the
          ring is carried to wherever that object is, seen from the text (on its plane), and turns about that
  wobble  per letter instance, before the letters are realised: a rise and a sway on a wave that travels along the word
          and a turn about each letter's own middle with the wave's slope, scaled by the keyed `WobAmp`, the wave at the
          keyed `WobPhase`

Everything that changes is a key on a Value node (`Orbit`, `WobAmp`, `WobPhase`); nothing runs at render time."""
import math

import bpy

ORBIT, WOB_AMP, WOB_PHASE = "Orbit", "WobAmp", "WobPhase"


def _val(g, name, v):
    n = g.node("ShaderNodeValue", name=name)
    n.outputs[0].default_value = float(v)
    return n.outputs[0]


def _xyz(g, x, y, z):
    n = g.node("ShaderNodeCombineXYZ")
    for i, v in enumerate((x, y, z)):
        g.put(n.inputs[i], v)
    return n.outputs[0]


def _sep(g, vec):
    n = g.node("ShaderNodeSeparateXYZ")
    g.put(n.inputs[0], vec)
    return n.outputs[0], n.outputs[1], n.outputs[2]


def bend(g, P, geo):
    """geometry -> geometry in the word's own frame (before it is placed): its points bent round the circle of
    `kin.bend` through the pivot. The pivot is P["pivot"] less the placement (tx, ty)."""
    b = P["kin"]["bend"]
    R, s = float(b["radius"]), float(b["side"])
    px, py = P["pivot"][0] - P["tx"], P["pivot"][1] - P["ty"]
    x, y, z = _sep(g, g.node("GeometryNodeInputPosition").outputs[0])
    psi = g.math("DIVIDE", g.math("SUBTRACT", x, px), R)                   # the angle round the circle
    r = g.math("ADD", R, g.math("MULTIPLY", g.math("SUBTRACT", y, py), s))  # the radius: further out, or in, with y
    nx = g.math("ADD", px, g.math("MULTIPLY", r, g.math("SINE", psi)))
    ny = g.math("ADD", py, g.math("MULTIPLY", g.math("SUBTRACT", g.math("MULTIPLY", r, g.math("COSINE", psi)), R), s))
    sp = g.node("GeometryNodeSetPosition")
    g.put(sp.inputs["Geometry"], geo)
    g.put(sp.inputs["Position"], _xyz(g, nx, ny, z))
    return sp.outputs[0]


def orbit_value(g, kin):
    """The keyable `Orbit` Value node (degrees counter-clockwise) of a word with `orbit`, made once."""
    return _val(g, ORBIT, kin["orbit"]["keys"][0][1])


def orbit(g, P, geo, angle_deg):
    """geometry -> geometry: the placed word turned about its ring's centre by `angle_deg` (a socket, degrees
    counter-clockwise); with `track` the centre it is laid out round is carried to the tracked object."""
    o = P["kin"]["orbit"]
    cx, cy = o["center"]
    a = g.node("GeometryNodeTransform")                      # to the centre, turn, and back (or on to the tracked one)
    g.put(a.inputs["Geometry"], geo)
    a.inputs["Translation"].default_value = (-cx, -cy, 0.0)
    b = g.node("GeometryNodeTransform")
    g.put(b.inputs["Geometry"], a.outputs[0])
    g.put(b.inputs["Rotation"], _xyz(g, 0.0, 0.0, g.math("RADIANS", angle_deg)))
    if o.get("track"):                                       # the object's place seen from the text, on the text's plane
        info = g.node("GeometryNodeObjectInfo", transform_space="RELATIVE")
        info.inputs["Object"].default_value = bpy.data.objects[o["track"]]
        tx, ty, _ = _sep(g, info.outputs["Location"])
        g.put(b.inputs["Translation"], _xyz(g, tx, ty, 0.0))
    else:
        b.inputs["Translation"].default_value = (cx, cy, 0.0)
    return b.outputs[0]


def wobble(g, P, inst):
    """Character instances -> instances: each letter risen, swayed and turned about its own pivot (g.piv, the String to
    Curves pivot of each character, in its local space) on the travelling wave of mkmmd.core.typewobble.letter."""
    w = P["kin"]["wobble"]
    em = P["em"]
    amp = _val(g, WOB_AMP, w["amp"][0][1])
    phase = _val(g, WOB_PHASE, w["phase"][0][1])
    idx = g.node("GeometryNodeInputIndex").outputs[0]
    th = g.math("MULTIPLY", g.math("SUBTRACT", phase, g.math("DIVIDE", idx, float(w["wave"]))), 2.0 * math.pi)
    s, c = g.math("SINE", th), g.math("COSINE", th)
    lift = g.math("MULTIPLY", amp, float(w["amount"]) * em)
    rz = g.math("MULTIPLY", g.math("MULTIPLY", amp, c), -math.radians(float(w["tilt"])))
    rot = g.node("GeometryNodeRotateInstances")
    g.put(rot.inputs["Instances"], inst)
    g.put(rot.inputs["Rotation"], _xyz(g, 0.0, 0.0, rz))
    g.put(rot.inputs["Pivot Point"], g.piv)
    rot.inputs["Local Space"].default_value = True
    tr = g.node("GeometryNodeTranslateInstances")
    g.put(tr.inputs["Instances"], rot.outputs[0])
    g.put(tr.inputs["Translation"], _xyz(g, g.math("MULTIPLY", g.math("MULTIPLY", lift, c), float(w["sway"])),
                                         g.math("MULTIPLY", lift, s), 0.0))
    tr.inputs["Local Space"].default_value = False
    return tr.outputs[0]


def key(ctx, nt, kin, key_node):
    """Key the ring's turn and the wobble's amplitude and phase onto their Value nodes (`key_node` is the stage's keyer);
    returns how many keys were written."""
    n = 0
    if kin.get("orbit"):
        key_node(ctx, nt, ORBIT, kin["orbit"]["keys"])
        n += len(kin["orbit"]["keys"])
    if kin.get("wobble"):
        key_node(ctx, nt, WOB_AMP, kin["wobble"]["amp"])
        key_node(ctx, nt, WOB_PHASE, kin["wobble"]["phase"])
        n += len(kin["wobble"]["amp"]) + len(kin["wobble"]["phase"])
    return n
