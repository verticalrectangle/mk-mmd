"""hand_model: one hand's rest geometry for the grip solvers (mkmmd/solvers/grip.py), written to an .npz. The build's
pose stage calls `hand_arrays(arm, side, radius)` for the same arrays inside its own Blender session.

args:
  armature  armature name, or null for the scene's only MMD armature
  side      "L" | "R"
  radius    keep skin vertices within this distance (m) of the wrist head (default 0.16)
  frame     Blender frame at which the armature's world matrix is read (default: the current frame)
  out       path of the .npz to write
The hand chain is the wrist followed by the five finger chains of core.bonemap.FINGERS, four entries each (thumb0 thumb1
thumb2 tip, index1 index2 index3 tip, ...). A tip bone the model lacks becomes a virtual entry (name ""): it ends the
finger where the skin weighted to the last real bone ends, and is never keyed.
npz keys, all in world space at `frame`, rest pose, metres (n = 21 chain entries):
  side           "L" | "R"
  names, sem     n Blender bone names ("" = virtual) and their semantic names
  parents        n indices into the chain, -1 = parent outside it
  heads, tails   (n, 3) bone heads and tails
  rest_rot       (n, 3, 3) world rest orientation of every bone (columns = the bone's x y z axes)
  V              (m, 3) rest-pose skin vertices near the hand
  W              (m, n + 1) their normalised weights on the chain; the last column is everything else, which stays put
  arm_world      (4, 4) the armature's world matrix at `frame`
  arm_chain      (3, 3) heads of the arm, elbow and wrist bones (NaN where the model lacks one)
  forearm        names of the forearm bones whose skin was kept
  meshes, vertex names of the meshes the skin came from and (m, 2) [index into meshes, vertex index] of every row of V
Only vertex groups named after bones count: mmd_tools also keeps non-bone data (mmd_edge_scale, ...) as groups."""
import bpy
import numpy as np

from ..core import bonemap
from . import scene as S
from .runtime import op

FOREARM = ("elbow", "arm_twist", "arm_twist1", "arm_twist2", "arm_twist3", "wrist_twist", "wrist_twist1",
           "wrist_twist2", "wrist_twist3")
DEFAULT_RADIUS = 0.16


def _unit(v):
    return v / np.linalg.norm(v)


def chain_layout(smap, side):
    """[(semantic name, Blender name or None)] for the wrist and the five fingers; None marks a missing tip bone.
    Raises KeyError listing every required bone the model lacks."""
    out, missing = [], []
    wrist = smap.get(f"wrist.{side}")
    if wrist is None:
        missing.append(f"wrist.{side}")
    out.append((f"wrist.{side}", wrist))
    for stems in bonemap.FINGERS.values():
        for k, stem in enumerate(stems):
            sem = f"{stem}.{side}"
            name = smap.get(sem)
            if name is None and k < 3:
                missing.append(sem)
            out.append((sem, name))
    if missing:
        raise KeyError(f"the {side} hand needs these bones, which the model lacks: {', '.join(missing)}")
    return out


def _skin(arm, meshes, groups, head, radius, dg):
    """The mesh vertices within radius of `head` that are weighted to at least one bone named in `groups`: their rest
    positions in world space (m, 3), per vertex {bone name: weight} over every bone it is weighted to, the names of
    the meshes that contributed, and (m, 2) [index into those names, vertex index]."""
    bones = arm.data.bones
    pts, rows, used, ids = [], [], [], []
    for mesh in meshes:
        me = mesh.data
        n = len(me.vertices)
        if not n:
            continue
        co = np.empty(n * 3)
        me.vertices.foreach_get("co", co)
        mm = np.array([list(r) for r in mesh.evaluated_get(dg).matrix_world])
        world = co.reshape(n, 3) @ mm[:3, :3].T + mm[:3, 3]
        near = np.where(np.linalg.norm(world - head, axis=1) <= radius)[0]
        if not len(near):
            continue
        gi = {g.index: g.name for g in mesh.vertex_groups if g.name in bones}
        kept = 0
        for i in near:
            ws = {}
            for g in me.vertices[int(i)].groups:
                name = gi.get(g.group)
                if name is not None and g.weight > 0:
                    ws[name] = ws.get(name, 0.0) + g.weight
            if ws and any(nm in groups for nm in ws):
                pts.append(world[i])
                rows.append(ws)
                ids.append((len(used), int(i)))
                kept += 1
        if kept:
            used.append(mesh.name)
    return np.array(pts, float).reshape(-1, 3), rows, used, np.array(ids, int).reshape(-1, 2)


def hand_arrays(arm, side, radius=DEFAULT_RADIUS):
    """The arrays of one hand of `arm` (the module docstring lists them) at the scene's current frame, for the grip
    solvers. Usable inside a running build: it reads the rest bones, the rest mesh and the armature's current
    world matrix, never the pose."""
    side = str(side).upper()
    if side not in ("L", "R"):
        raise ValueError(f"side must be L or R, got {side!r}")
    radius = float(radius or DEFAULT_RADIUS)
    dg = bpy.context.evaluated_depsgraph_get()
    mw = np.array([list(r) for r in arm.evaluated_get(dg).matrix_world])
    smap = S.semantic_map(arm)
    layout = chain_layout(smap, side)
    bones = arm.data.bones
    n = len(layout)
    real = [i for i, (_s, name) in enumerate(layout) if name is not None]
    index = {layout[i][1]: i for i in real}
    R3 = mw[:3, :3]
    heads, tails = np.zeros((n, 3)), np.zeros((n, 3))
    rest_rot = np.tile(np.eye(3), (n, 1, 1))
    parents = np.full(n, -1, int)
    for i in real:
        b = bones[layout[i][1]]
        heads[i] = mw[:3, :3] @ np.array(b.head_local) + mw[:3, 3]
        tails[i] = mw[:3, :3] @ np.array(b.tail_local) + mw[:3, 3]
        m = R3 @ np.array([list(r) for r in b.matrix_local])[:3, :3]
        rest_rot[i] = m / np.linalg.norm(m, axis=0, keepdims=True)
        if b.parent is not None:
            parents[i] = index.get(b.parent.name, -1)
    fore = [smap[f"{s}.{side}"] for s in FOREARM if f"{s}.{side}" in smap]
    wanted = set(index) | set(fore)
    V, rows, used, vertex = _skin(arm, S.model_meshes(arm), wanted, heads[0], radius, dg)
    if not len(V):
        raise ValueError(f"no skin vertices within {radius} m of {layout[0][1]}: is the model's mesh skinned to "
                         f"{arm.name}?")
    W = np.zeros((len(V), n + 1))
    for r, ws in enumerate(rows):
        tot = sum(ws.values())
        for name, w in ws.items():
            W[r, index.get(name, n)] += w / tot
    virtual = [i for i in range(n) if layout[i][1] is None]
    dom = np.argmax(W, axis=1)
    for i in virtual:                      # a missing tip bone: continue the finger to the end of its last bone's skin
        p = i - 1
        prev = heads[p] - heads[p - 1]
        d = _unit(prev)
        s = (V[dom == p] - heads[p]) @ d
        ln = float(np.clip(s.max() if len(s) else 0.6 * np.linalg.norm(prev), 0.4 * np.linalg.norm(prev),
                           1.4 * np.linalg.norm(prev)))
        heads[i] = heads[p] + d * ln
        tails[i] = heads[i] + d * 0.3 * ln
        rest_rot[i] = rest_rot[p]
        parents[i] = p
    arm_chain = np.full((3, 3), np.nan)
    for k, stem in enumerate(("arm", "elbow", "wrist")):
        name = smap.get(f"{stem}.{side}")
        if name is not None:
            arm_chain[k] = mw[:3, :3] @ np.array(bones[name].head_local) + mw[:3, 3]
    return {"side": np.array(side), "names": np.array([name or "" for _s, name in layout]),
            "sem": np.array([s for s, _n in layout]), "parents": parents, "heads": heads, "tails": tails,
            "rest_rot": rest_rot, "V": V, "W": W, "arm_world": mw, "arm_chain": arm_chain,
            "forearm": np.array(fore, dtype=str), "vertex": vertex, "meshes": np.array(used, dtype=str)}


@op("hand_model")
def hand_model(args):
    arm = S.find_armature(args.get("armature") or None)
    sc = bpy.context.scene
    keep_frame = sc.frame_current
    if args.get("frame") is not None:
        sc.frame_set(int(args["frame"]))
    try:
        a = hand_arrays(arm, args.get("side", "R"), args.get("radius"))
    finally:
        if args.get("frame") is not None:
            sc.frame_set(keep_frame)
    np.savez(args["out"], **a)
    mid = 1 + 4 * list(bonemap.FINGERS).index("middle") + 3
    return {"out": args["out"], "armature": arm.name, "side": str(a["side"]), "frame": args.get("frame"),
            "bones": len(a["names"]), "names": [str(x) for x in a["names"]],
            "virtual": [str(s) for s, nm in zip(a["sem"], a["names"]) if not nm], "vertices": int(len(a["V"])),
            "meshes": [str(x) for x in a["meshes"]], "forearm": [str(x) for x in a["forearm"]], "wrist": a["heads"][0],
            "radius": float(args.get("radius") or DEFAULT_RADIUS),
            "hand_length": float(np.linalg.norm(a["heads"][mid] - a["heads"][0]))}
