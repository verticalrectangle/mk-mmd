"""Skin weights of the body part: analytic, smooth, no per-vertex painting.

Every shell skins along its own chain of bones with smooth blends centred on the joints (`chain_weights`): the torso
along the spine by height (plus the shoulder girdle), an arm along 腕 > 腕捩 > ひじ > 手捩 > 手首 (the twist bones take
a long ramp so a twist never collapses the skin), a leg along 足 > ひざ > 足首 (the pelvis end blends with 下半身) and
every finger along its three phalanges (roots blend with 手首). `bones` maps bone names to their head positions
(from the skeleton); sided names are built from the side prefix."""
import numpy as np

from .body_geom import smoothstep, unit

SIDE_JP = {"L": "左", "R": "右"}


def chain_weights(s, joints, widths):
    """(n, k+1) weights of k+1 consecutive bones along the parameter s: bone j hands over to bone j+1 around
    joints[j] with half-width widths[j] (smoothstep). Rows sum to one; overlapping blends are made monotone."""
    s = np.asarray(s, float)
    k = len(joints)
    a = np.zeros((len(s), k + 2))                       # a[:, j]: share of the vertex still on bones 0..j-1 (a_0 = 0)
    a[:, k + 1] = 1.0
    prev = np.zeros(len(s))
    for j in range(k):
        h = max(float(widths[j]), 1e-6)
        aj = 1.0 - smoothstep((s - (joints[j] - h)) / (2.0 * h))     # 1 before the joint, 0 after it
        aj = np.maximum(aj, prev)
        a[:, j + 1] = aj
        prev = aj
    return a[:, 1:] - a[:, :-1]


def _add(W, name, w):
    if name in W:
        W[name] = W[name] + w
    else:
        W[name] = np.array(w, float)


def spine_weights(shape, z, x, bones):
    """Torso weights by height and by lateral position (shoulder girdle, thighs). Returns {bone: (n,)}."""
    L = shape.land
    n = len(z)
    ub, ub2, neck = L["upper_body"][2], L["upper_body2"][2], L["neck"][2]
    top = neck - 0.0185                                           # 首 is 100 % from just below the seam ring
    chain = ["下半身", "上半身", "上半身2", "首"]
    joints = [0.5 * (L["lower_body"][2] + ub) + 0.004, ub2 - 0.002, top - 0.012]
    widths = [0.030, 0.035, 0.016]
    W4 = chain_weights(z, joints, widths)
    W = {b: W4[:, i].copy() for i, b in enumerate(chain)}
    for side in ("L", "R"):
        sx = 1.0 if side == "L" else -1.0
        jp = SIDE_JP[side]
        on_side = x * sx > 0.0
        u = np.abs(x)
        # shoulder girdle: 肩 takes the trapezius and the top of the shoulder, 腕 a share near the arm root
        sh, ar = L[f"shoulder.{side}"], L[f"arm.{side}"]
        f_lat = smoothstep((u - (abs(sh[0]) - 0.012)) / max(abs(ar[0]) - abs(sh[0]) + 0.012, 1e-3))
        f_z = smoothstep((z - (ub2 + 0.03)) / 0.06)
        away = smoothstep((u - 0.030) / 0.025)                      # nothing of the girdle on the neck itself
        f_sh = 0.80 * f_lat * f_z * away
        d = np.linalg.norm(np.stack([u - abs(ar[0]), z - ar[2]], 1), axis=1)
        f_arm = 0.75 * np.exp(-(d / 0.066) ** 2) * away
        tot = f_sh + f_arm
        take = np.clip(tot, 0.0, 0.92)
        share = np.where(tot > 1e-9, f_sh / np.maximum(tot, 1e-9), 0.0)
        take = np.where(on_side, take, 0.0)
        for b in list(W):
            W[b] = W[b] * (1.0 - take)
        _add(W, f"{jp}肩", take * share)
        _add(W, f"{jp}腕", take * (1.0 - share))
        # the pelvis splits into the thighs below the crotch line
        leg = L[f"leg.{side}"]
        f_z = smoothstep((leg[2] + 0.045 - z) / 0.075)
        f_x = smoothstep((u - 0.012) / 0.040)
        take = np.where(on_side, 0.95 * f_z * f_x, 0.0)
        for b in list(W):
            W[b] = W[b] * (1.0 - take)
        _add(W, f"{jp}足", take)
    return W


FINGER_JP = {"index": "人指", "middle": "中指", "ring": "薬指", "little": "小指"}
KNUCKLE = 0.0075                          # half-width (m) of the 手首 -> first phalanx blend around each knuckle
FINGER_ORDER = ("index", "middle", "ring", "little")


def arm_weights(shape, sh, side):
    """Weights of an arm shell (s = arclength from the shoulder joint, negative inside the torso). Past the knuckle line
    the palm's 32-point ring splits into four finger blocks: block f follows that finger's first bone."""
    info = sh.info
    la, lf = info["la"], info["lf"]
    jp = SIDE_JP[side]
    s = sh.s
    chain = [f"{jp}腕", f"{jp}腕捩", f"{jp}ひじ", f"{jp}手捩", f"{jp}手首"]
    joints = [0.40 * la, la, la + 0.40 * lf, la + lf]
    widths = [0.30 * la, 0.030, 0.30 * lf, 0.016]
    Wm = chain_weights(s, joints, widths)
    W = {b: Wm[:, i].copy() for i, b in enumerate(chain)}
    # the root follows the shoulder girdle (肩) and, on the underarm side, the chest (上半身2): the deltoid keeps 腕
    sx = 1.0 if side == "L" else -1.0
    root = 1.0 - smoothstep((s + 0.020) / 0.090)                     # 1 inside the torso, 0 past s = 0.07
    inner = smoothstep((abs(shape.land[f"arm.{side}"][0]) - np.abs(sh.verts[:, 0])) / 0.030)
    low = smoothstep((shape.land[f"arm.{side}"][2] - sh.verts[:, 2] + 0.012) / 0.030)
    f_sh = 0.45 * root
    f_ch = 0.62 * root * np.maximum(inner, 0.6 * low)
    for b in list(W):
        W[b] = W[b] * (1.0 - f_sh - f_ch)
    _add(W, f"{jp}肩", f_sh)
    _add(W, "上半身2", f_ch)
    # the finger bases
    h = KNUCKLE
    b = smoothstep((s - (info["s_mcp"] - h)) / (2.0 * h))
    M = 32
    k = np.floor(np.nan_to_num(sh.theta) / (2 * np.pi / M)).astype(int) % M
    fi = np.where(k < 16, k // 4, (31 - k) // 4)
    wrist = f"{jp}手首"
    base = W[wrist].copy()
    W[wrist] = base * (1.0 - b)
    for j, fname in enumerate(FINGER_ORDER):
        _add(W, f"{jp}{FINGER_JP[fname]}１", base * b * (fi == j))
    return W


def leg_weights(shape, sh, side):
    """Weights of a leg shell (s = arclength from the hip joint, negative inside the pelvis)."""
    info = sh.info
    jp = SIDE_JP[side]
    s = sh.s
    chain = [f"{jp}足", f"{jp}ひざ", f"{jp}足首"]
    Wm = chain_weights(s, [info["sK"], info["sA"] - 0.002], [0.045, 0.030])
    W = {b: Wm[:, i].copy() for i, b in enumerate(chain)}
    w_lb = 1.0 - smoothstep((s + 0.020) / 0.075)                  # the pelvis end belongs to the lower body
    for b in list(W):
        W[b] = W[b] * (1.0 - w_lb)
    _add(W, "下半身", w_lb)
    return W


def finger_weights(shape, sh, side):
    info = sh.info
    jp = SIDE_JP[side]
    f = info["finger"]
    s = sh.s
    s1, s2, s3 = info["s"][1], info["s"][2], info["s"][3]
    if f == "thumb":
        chain = [f"{jp}手首", f"{jp}親指０", f"{jp}親指１", f"{jp}親指２"]
        joints = [s1 - 0.002, s2, s3]
        widths = [0.012, 0.0075, 0.0060]
    else:
        names = FINGER_JP[f]
        chain = [f"{jp}手首", f"{jp}{names}１", f"{jp}{names}２", f"{jp}{names}３"]
        joints = [-info["web"], s2, s3]
        widths = [KNUCKLE, 0.0065, 0.0055]
    Wm = chain_weights(s, joints, widths)
    return {b: Wm[:, i].copy() for i, b in enumerate(chain)}


def body_weights(shape, shells, ranges, gidx, nverts, bones):
    """{bone name: (nverts,) weights} over the joined body mesh. `bones` is the set of deforming bone names; shells
    write only the vertices they own (welded vertices belong to the shell that stores them)."""
    out = {}

    def put(sh, W):
        own = np.ones(len(sh), bool)
        for i in sh.ext:
            own[i] = False
        g = gidx[sh.name][own]
        for bone, w in W.items():
            if bone not in bones:
                continue
            arr = out.setdefault(bone, np.zeros(nverts))
            arr[g] += np.asarray(w)[own]

    for sh in shells:
        kind = sh.info["kind"]
        side = sh.name[-1] if sh.name[-2:] in ("_L", "_R") else None
        if kind == "torso":
            put(sh, spine_weights(shape, sh.verts[:, 2], sh.verts[:, 0], bones))
        elif kind == "arm":
            put(sh, arm_weights(shape, sh, side))
        elif kind == "leg":
            put(sh, leg_weights(shape, sh, side))
        elif kind == "finger":
            put(sh, finger_weights(shape, sh, side))
    tot = np.zeros(nverts)
    for w in out.values():
        tot += w
    missing = tot < 1e-9
    if missing.any():
        raise ValueError(f"{int(missing.sum())} body vertices without weights")
    return cap_weights({b: w / np.maximum(tot, 1e-12) for b, w in out.items()}, 4)


def cap_weights(W, k=4, floor=1e-4):
    """Keep the k largest weights of every vertex and renormalise (PMX skinning holds four bones); weights below
    `floor` are dropped; bones left without any weight are removed from the dict."""
    names = list(W)
    M = np.stack([W[n] for n in names], 1)
    if M.shape[1] > k:
        drop = np.argsort(-M, axis=1)[:, k:]
        np.put_along_axis(M, drop, 0.0, axis=1)
    M[M < floor] = 0.0
    M /= np.maximum(M.sum(1, keepdims=True), 1e-12)
    return {n: M[:, i].copy() for i, n in enumerate(names) if M[:, i].any()}
