"""Surface of the body part: torso + neck (one loft up to the neck seam ring), an arm with its palm (one loft), a leg with
its foot (one bent loft), and the five fingers of each hand (tubes rooted inside the palm). Left side built, right side
mirrored. Sizes come from `Shape.dims` (torso cuts, limb diameters, foot) and `Shape.land`.

Shell names: torso, arm_L, leg_L, <finger>_L (index middle ring little thumb), and the same with _R.
Atlas tiles (u0, v0, u1, v1, v up) place every shell in the skin texture."""
import numpy as np

from . import body_geom as G
from .body_geom import Path, Table, Shell, tube, mirror_x, unit, pchip

ATLAS = {
    "torso": (0.00, 0.50, 0.50, 1.00),
    "arm": (0.50, 0.50, 0.75, 1.00),
    "leg": (0.75, 0.50, 1.00, 1.00),
    "index": (0.00, 0.25, 0.20, 0.50),
    "middle": (0.20, 0.25, 0.40, 0.50),
    "ring": (0.40, 0.25, 0.60, 0.50),
    "little": (0.60, 0.25, 0.80, 0.50),
    "thumb": (0.80, 0.25, 1.00, 0.50),
}
FINGERS = ("index", "middle", "ring", "little")


def _seg(a, b):
    return float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))


def _half(pair):
    return 0.5 * float(pair[0]), 0.5 * float(pair[1])


# ---------------------------------------------------------------- torso + neck
def torso_rows(shape):
    """(z, half width, front y, back y, n) rows of the torso cuts: the pelvis tapers into the crotch below the hips (the
    cuts include the thighs there), and extra rows refine the shoulder line where it turns into the neck. The last row is
    the neck seam ring (a plain ellipse)."""
    T = shape.dims["torso"]
    z = np.array(T["z"], float)
    hw = np.array(T["width"], float) / 2
    yf = np.array(T["y_front"], float)
    yb = np.array(T["y_back"], float)
    nz = pchip([z[0] - 0.08, 0.80, 0.93, 1.03, 1.10, 1.14, z[-1]], [2.4, 2.5, 2.25, 2.3, 2.5, 2.3, 2.0])
    # below the widest cut the thighs carry the silhouette: taper the pelvis into the crotch between them
    z_hip = float(z[int(np.argmax(hw))])
    keep = z >= z_hip - 1e-9
    z, hw, yf, yb = z[keep], hw[keep], yf[keep], yb[keep]
    low = [(z_hip - 0.020, 0.0985, -0.094, 0.052), (z_hip - 0.045, 0.082, -0.083, 0.046),
           (z_hip - 0.070, 0.062, -0.071, 0.038), (z_hip - 0.092, 0.040, -0.056, 0.030)]
    zl = np.array([r[0] for r in low][::-1])
    z = np.concatenate([zl, z])
    hw = np.concatenate([[r[1] for r in low][::-1], hw])
    yf = np.concatenate([[r[2] for r in low][::-1], yf])
    yb = np.concatenate([[r[3] for r in low][::-1], yb])
    # the shoulder joint must sit inside the torso volume (the cuts exclude the arms): fuller trapezius/deltoid line
    zt = z[-1]
    sh_x = abs(float(shape.land["arm.L"][0]))
    sh_z = float(shape.land["arm.L"][2])
    line = pchip([sh_z - 0.040, sh_z - 0.020, sh_z, sh_z + 0.019, sh_z + 0.0255, sh_z + 0.032, zt],
                 [0.0838, 0.0805, 0.0735, 0.0600, 0.0470, 0.0345, hw[-1]])
    hw = np.where(z >= sh_z - 0.040, np.maximum(hw, line(np.clip(z, sh_z - 0.040, zt))), hw)
    # the bust's upper edge retreats gradually instead of in one step
    k = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    for _ in range(2):
        ypad = np.concatenate([[yf[0]] * 2, yf, [yf[-1]] * 2])
        sm = np.convolve(ypad, k, mode="valid")
        sel = (z > 0.97) & (z < zt - 0.05)
        yf = np.where(sel, sm, yf)
    extra = [z[-1] - 0.0045, z[-1] - 0.0105, z[-1] - 0.0295, z[-1] - 0.0495, z[-1] - 0.068]
    zz = np.array(sorted(set(np.round(list(z) + extra, 6))))
    f = pchip(z, np.stack([hw, yf, yb], -1))
    v = f(zz)
    return [(float(a), float(b), float(c), float(d), float(nz(a))) for a, (b, c, d) in zip(zz, v)]


def neck_top(shape):
    """The neck seam ring description published to the head: z, centre (x, y), rx (half width), ry (half depth)."""
    rows = torso_rows(shape)
    z, hw, yf, yb, _ = rows[-1]
    return dict(z=z, center=(0.0, 0.5 * (yf + yb)), rx=hw, ry=0.5 * (yb - yf))


def torso_shell(shape, M=32):
    rows = torso_rows(shape)
    z = np.array([r[0] for r in rows])
    hw = np.array([r[1] for r in rows])
    yf = np.array([r[2] for r in rows])
    yb = np.array([r[3] for r in rows])
    nn = np.array([r[4] for r in rows])
    # a gentle sternum dip between the bust lobes: the front of the cuts is the lobes' extreme, so pull the centre back
    bust = np.exp(-((z - 1.03) / 0.045) ** 2) * 0.010
    tab = Table(z, rx=(yb - yf) / 2, ry=hw, ox=-(yb + yf) / 2, oy=np.zeros_like(z), n=nn)
    ss = z - z[0]
    path = Path([[0.0, 0.0, z[0]], [0.0, 0.0, z[-1]]], blend=0.0)
    sh = tube("torso", path, ss, lambda s: tab(s + z[0]), M, (0.0, -1.0, 0.0), cap0="dome", cap1="open", cap_rings=4,
              uv_rect=ATLAS["torso"], cap_len=(0.30, 1.0))
    sh.s = sh.s + z[0]                                          # the torso's parameter is the height itself
    sh.s_range = (float(z[0]), float(z[-1]))                    # v of the atlas tile = (z - z[0]) / (z[-1] - z[0])
    # sternum dip: move the front-centre vertices of the bust rows back
    zq = sh.verts[:, 2]
    th = sh.theta
    front = np.isfinite(th) & (np.cos(np.nan_to_num(th)) > 0.0)
    dip = np.interp(zq, z, bust) * np.where(front, np.cos(np.nan_to_num(th)) ** 6, 0.0) * (zq < z[-1] - 0.04)
    sh.verts = sh.verts + np.stack([np.zeros_like(dip), dip, np.zeros_like(dip)], 1)
    sh.info = dict(kind="torso")
    return sh


# ---------------------------------------------------------------- arm + palm + fingers (one welded surface)
def hand_cfg(shape):
    d = dict(web=0.010, gap=0.0013, thenar=0.0085, thenar_radius=0.022, slim=1.0, palm_slim=1.0, knuckle=0.0020,
             hypothenar=0.0040, pads=0.0024, pad_tip=0.0008)
    d.update(shape.dims.get("hand", {}))
    return d


def finger_shell(shape, name, rx_web=None, M=8):
    """A finger tube. The four fingers start at the web ring (`web` metres beyond the knuckle) with 8-point rings
    (4 palm side + 4 back, ring point j at angle (j + 1/2) * 45 degrees), so they weld to the palm's 32-point ring."""
    L = shape.land
    D = shape.dims["limb"]
    H = hand_cfg(shape)
    if name == "thumb":
        joints = [L[f"thumb{i}.L"] for i in (0, 1, 2)] + [L["thumb_tip.L"]]
        w, t = D["finger_thumb"]
        back = 0.016
        scale = [1.00, 1.40, 1.30, 1.14, 1.04, 0.95, 0.80, 0.62]
        M = 12
        J1, J2, J3, T = joints
        a_, r_, n_ = shape.frame
        root = J1 - unit(J2 - J1) * back + r_ * 0.011 - n_ * 0.004      # rooted inside the palm, towards its centre
        path = Path([root, J1, J2, J3, T], blend=0.004)
        s1 = _seg(root, J1)
        s2 = s1 + _seg(J1, J2)
        s3 = s2 + _seg(J2, J3)
        s4 = s3 + _seg(J3, T)
        knots = [0.0, s1, s1 + 0.5 * (s2 - s1), s2, s2 + 0.5 * (s3 - s2), s3, s3 + 0.6 * (s4 - s3), s4]
        ss = [0.0, 0.5 * s1, s1, s1 + 0.5 * (s2 - s1), s2 - 0.006, s2, s2 + 0.006, s2 + 0.5 * (s3 - s2), s3 - 0.005, s3,
              s3 + 0.005, s3 + 0.55 * (s4 - s3), s4]
        ref = unit(np.cross(unit(J3 - J1), n_))
        cap0, theta0 = "flat", 0.0
    else:
        joints = [L[f"{name}{i}.L"] for i in (1, 2, 3)] + [L[f"{name}_tip.L"]]
        w, t = D["finger_little"] if name == "little" else D["finger"]
        scale = [1.0, 0.99, 0.98, 0.97, 0.89, 0.88, 0.78, 0.64]
        J1, J2, J3, T = joints
        p_web = J1 + unit(J2 - J1) * H["web"]
        path = Path([p_web, J2, J3, T], blend=0.004)
        s1 = 0.0
        s2 = _seg(p_web, J2)
        s3 = s2 + _seg(J2, J3)
        s4 = s3 + _seg(J3, T)
        knots = [0.0, 0.5 * s2, s2 - 0.005, s2, s2 + 0.5 * (s3 - s2), s3, s3 + 0.6 * (s4 - s3), s4]
        ss = [0.0, 0.5 * s2, s2 - 0.006, s2, s2 + 0.006, s2 + 0.5 * (s3 - s2), s3 - 0.005, s3, s3 + 0.005,
              s3 + 0.55 * (s4 - s3), s4]
        ref = (0.0, -1.0, 0.0)
        cap0, theta0 = "open", np.pi / 8
    rx0, ry0 = 0.5 * w * H["slim"], 0.5 * t * H["slim"]
    sc = np.array(scale)
    rx = rx0 * sc
    if rx_web is not None:
        rx[0] = min(rx[0], rx_web)
        rx[1] = min(rx[1], 0.5 * (rx[0] + rx[2]) if rx[0] < rx[2] else rx[1])
    # fingertip pad: a little fuller and shifted to the palm side over the last phalanx
    pad = np.array([1, 1, 1, 1, 1.0, 1.05, 1.12, 1.16]) if name != "thumb" else np.array([1, 1, 1, 1, 1, 1.03, 1.08, 1.10])
    oy = np.array([0, 0, 0, 0, 0, 0.0002, 0.5, 0.35]) * np.array([1, 1, 1, 1, 1, 1, H["pad_tip"], H["pad_tip"]])
    tab = Table(knots, rx=rx, ry=ry0 * sc * pad, ox=np.zeros(8), oy=oy, n=np.full(8, 2.4 if name != "thumb" else 2.2))
    cap_len = 1.12 if name != "thumb" else 1.10
    sh = tube(f"{name}_L", path, ss, tab, M, ref, cap0=cap0, cap1="dome", cap_rings=3, theta0=theta0,
              uv_rect=ATLAS[name], cap_len=(1.0, cap_len))
    sh.info = dict(kind="finger", finger=name, path=path, s=[0.0, s1, s2, s3, s4], joints=joints,
                   welded=(name != "thumb"), web=H["web"], tab=tab, ref=ref, cap_len=cap_len)
    return sh


def nail_shell(shape, fsh, nu=7, lift=0.0004):
    """A nail plate on the back of a fingertip: a quad grid on the finger's surface (rows along the last phalanx, then over
    the first part of the tip dome), lifted by `lift` metres, rounded at the cuticle and at the free edge."""
    i = fsh.info
    s3, s4 = i["s"][3], i["s"][4]
    ld = s4 - s3
    frac = [0.00, 0.03, 0.07, 0.13, 0.22, 0.38, 0.55, 0.72, 0.86]           # along the plate on the phalanx
    gs = [0.30, 0.58, 0.80, 0.93, 1.00, 1.00, 1.00, 0.98, 0.94]             # half-width share of each row
    doms, gd = (0.40, 0.80, 1.12), (0.85, 0.62, 0.34)                      # rows over the tip dome
    s_a = s3 + 0.16 * ld
    rows = [("s", s_a + f * (s4 - s_a), g) for f, g in zip(frac, gs)] + [("d", a, g) for a, g in zip(doms, gd)]
    th_c = 0.5 * np.pi if i["finger"] == "thumb" else 1.5 * np.pi       # the ring angle of the back of the finger
    us = np.linspace(-1.0, 1.0, nu)
    tab, path, ref = i["tab"], i["path"], i["ref"]
    sec4 = tab(np.array([s4]))
    rx4, ry4 = float(sec4["rx"][0]), float(sec4["ry"][0])
    c4, t4 = path.point(np.array([s4]))[0], path.tangent(np.array([s4]))[0]
    rad = 0.5 * (rx4 + ry4) * i.get("cap_len", 1.25)                       # dome depth (cap_len of the finger tube)
    P = np.zeros((len(rows), nu, 3))
    N = np.zeros((len(rows), nu, 3))
    sv = []
    for r, (kind, v, g) in enumerate(rows):
        if kind == "s":
            sec = tab(np.array([v]))
            rxs, rys = float(sec["rx"][0]), float(sec["ry"][0])
            half = 0.5 * 0.60 * 2.0 * rxs * g / max(0.5 * (rxs + rys), 1e-4)
            p, n = G.surface_point(path, np.full(nu, v), tab, ref, th_c + us * half)
            P[r], N[r] = p, n
            sv.append(v)
        else:
            half = 0.5 * 0.60 * 2.0 * rx4 * g / max(0.5 * (rx4 + ry4), 1e-4)
            p4, n4 = G.surface_point(path, np.full(nu, s4), tab, ref, th_c + us * half)
            ax = c4 + t4 * rad * np.sin(v)
            P[r] = c4 + (p4 - c4) * np.cos(v) + t4 * rad * np.sin(v)
            N[r] = G.unit(P[r] - ax)
            sv.append(s4 + rad * np.sin(v))
    verts = (P + N * lift).reshape(-1, 3)
    nr = len(rows)
    faces, uvs = [], []
    for r in range(nr - 1):
        for c in range(nu - 1):
            faces.append([r * nu + c, r * nu + c + 1, (r + 1) * nu + c + 1, (r + 1) * nu + c])
            uvs.append(np.array([[0.5, 0.5]] * 4))
    sh = Shell(f"nail_{fsh.name}")
    sh.verts = verts
    sh.faces = faces
    sh.uv = uvs
    sh.s = np.repeat(np.array(sv), nu)
    sh.theta = np.full(len(verts), np.nan)
    sh.ring = np.full(len(verts), -1)
    sh.face_mat = [1] * len(faces)
    n0 = np.cross(verts[faces[0][1]] - verts[faces[0][0]], verts[faces[0][3]] - verts[faces[0][0]])
    if n0 @ N[0, nu // 2] < 0:
        sh.flip()
    sh.info = dict(i, kind="finger", nail=True)
    return sh


def arm_hand_shells(shape, M=32):
    """[arm_L, index_L, middle_L, ring_L, little_L, thumb_L]: the arm tube runs through the palm to the web ring, where
    the four fingers continue it (shared vertices) and 3 quads close the slits between them."""
    L = shape.land
    D = shape.dims["limb"]
    H = hand_cfg(shape)
    assert M == 32, "the palm ring is 4 fingers x (4 palm side + 4 back) points"
    # finger ring widths at the web: neighbours must not touch
    p_web = {f: L[f"{f}1.L"] + unit(L[f"{f}2.L"] - L[f"{f}1.L"]) * H["web"] for f in FINGERS}
    rx_nom = {f: 0.5 * (D["finger_little"][0] if f == "little" else D["finger"][0]) for f in FINGERS}
    rx_web = dict(rx_nom)
    for fa, fb in zip(FINGERS[:-1], FINGERS[1:]):
        space = abs(float(p_web[fb][1] - p_web[fa][1]))
        room = space - H["gap"]
        for f in (fa, fb):
            rx_web[f] = min(rx_web[f], 0.5 * room)
    fingers = {f: finger_shell(shape, f, rx_web[f]) for f in FINGERS}
    # target web ring: finger f owns palm-side points 4f..4f+3 and back points 28-4f..31-4f
    T = np.zeros((32, 3))
    amap = {}
    for fi, f in enumerate(FINGERS):
        ring = fingers[f].rings[0]
        for j in range(8):
            k = 4 * fi + j if j < 4 else 28 - 4 * fi + (j - 4)
            T[k] = ring[j]
            amap[(f, j)] = k
    J, E, W = L["arm.L"], L["elbow.L"], L["wrist.L"]
    Qw = T.mean(axis=0)
    path = Path([J, E, W, Qw], blend=0.035)
    la, lf = _seg(J, E), _seg(E, W)
    lp = path.length - la - lf
    s_w, s_web = la + lf, path.length
    top, mid, el = _half(D["upper_arm_top"]), _half(D["upper_arm_mid"]), _half(D["elbow"])
    ft, fm, wr, pk = _half(D["forearm_top"]), _half(D["forearm_mid"]), _half(D["wrist"]), _half(D["palm_knuckles"])
    pk = (pk[0] * H["palm_slim"], pk[1] * H["palm_slim"])
    ball = 1.30
    # the palm's end section is measured on the web ring so the blend to it is gentle
    t_end = unit(path.tangent(np.array([s_web]))[0])
    ef_end = unit(np.array([0.0, -1.0, 0.0]) - (np.array([0.0, -1.0, 0.0]) @ t_end) * t_end)
    eb_end = np.cross(t_end, ef_end)
    ya, yb = (T - Qw) @ ef_end, (T - Qw) @ eb_end
    rx_e, ry_e = 0.5 * (ya.max() - ya.min()), 0.5 * (yb.max() - yb.min())
    ox_e, oy_e = 0.5 * (ya.max() + ya.min()), 0.5 * (yb.max() + yb.min())
    rows = [
        (0.0, top[0] * ball, top[1] * ball, 0.0, 0.0, 2.0),
        (0.12 * la, top[0] * 1.13, top[1] * 1.13, 0.0, 0.0, 2.0),
        (0.30 * la, top[0] * 1.03, top[1] * 1.03, 0.0, 0.0, 2.0),
        (0.55 * la, mid[0], mid[1], 0.0, 0.0, 2.0),
        (0.85 * la, el[0] * 1.03, el[1] * 1.03, 0.0, 0.0, 2.0),
        (la, el[0], el[1], 0.0, 0.0, 2.05),
        (la + 0.22 * lf, ft[0], ft[1], 0.0, 0.0, 2.1),
        (la + 0.55 * lf, fm[0], fm[1], 0.0, 0.0, 2.2),
        (la + 0.85 * lf, 0.5 * (fm[0] + wr[0]), 0.5 * (fm[1] + wr[1]) * 0.95, 0.0, 0.0, 2.35),
        (s_w, wr[0], wr[1], 0.0, 0.0, 2.6),
        (s_w + 0.20 * lp, 0.5 * (wr[0] + pk[0]) * 1.03, 0.5 * (wr[1] + pk[1]) * 0.85, 0.2 * ox_e, 0.0, 2.8),
        (s_w + 0.50 * lp, pk[0] * 0.97, pk[1] * 0.80, 0.6 * ox_e, 0.0, 3.0),
        (s_w + 0.80 * lp, rx_e * 1.02, max(ry_e, pk[1] * 0.78), ox_e, oy_e, 3.0),
        (s_web, rx_e, ry_e, ox_e, oy_e, 3.0),
    ]
    R = np.array(rows)
    tab = Table(R[:, 0], rx=R[:, 1], ry=R[:, 2], ox=R[:, 3], oy=R[:, 4], n=R[:, 5])
    ss = [f * la for f in (0.0, 0.08, 0.18, 0.32, 0.46, 0.60, 0.74, 0.86, 0.93, 1.0)]
    ss += [la + f * lf for f in (0.07, 0.16, 0.30, 0.45, 0.60, 0.75, 0.88, 1.0)]
    ss += [s_w + f * lp for f in (0.12, 0.26, 0.40, 0.52, 0.63, 0.73, 0.81, 0.88, 0.94, 1.0)]
    arm = tube("arm_L", path, ss, tab, M, (0.0, -1.0, 0.0), cap0="dome", cap1="open", cap_rings=3,
               theta0=np.pi / M, uv_rect=ATLAS["arm"], cap_len=(1.0, 1.0))
    arm.orient_outward() if False else None
    # blend the palm rings into the web ring
    idx = arm.ring_index
    s_ring = np.array([arm.s[idx[i, 0]] for i in range(len(idx))])
    s0 = s_w + 0.40 * lp
    for i in range(len(idx)):
        if s_ring[i] <= s0:
            continue
        w_ = float(G.smoothstep((s_ring[i] - s0) / (s_web - s0)))
        arm.verts[idx[i]] = (1 - w_) * arm.verts[idx[i]] + w_ * T
    arm.rings = arm.verts[idx]
    # thenar eminence: a soft bump on the palm side at the thumb's root
    thumb = finger_shell(shape, "thumb")
    c = L["thumb0.L"] + shape.frame[2] * 0.004
    zone = (s_ring[0] * 0 + arm.s > s_w - 0.005) & (arm.s < s_w + 0.7 * lp) & np.isfinite(arm.theta)
    d = np.linalg.norm(arm.verts - c, axis=1)
    bump = H["thenar"] * np.exp(-(d / H["thenar_radius"]) ** 2) * zone
    ctr = np.array([path.point(arm.s[i]) for i in range(len(arm.verts))])
    out = G.unit(arm.verts - ctr)
    arm.verts = arm.verts + out * bump[:, None]
    # knuckle relief on the back of the hand, hypothenar and finger-base pads on the palm side
    a_h, r_h, n_h = shape.frame
    th_ = np.nan_to_num(arm.theta)
    dorsal = (th_ > np.pi) & (th_ < 2 * np.pi) & np.isfinite(arm.theta)
    palmar = (th_ > 0) & (th_ < np.pi) & np.isfinite(arm.theta)
    near_hand = arm.s > s_w - 0.01
    extra = np.zeros(len(arm.verts))                                  # outward displacement (m)
    for f in FINGERS:
        c = L[f"{f}1.L"] - n_h * 0.010
        extra += H["knuckle"] * np.exp(-(np.linalg.norm(arm.verts - c, axis=1) / 0.0105) ** 2) * dorsal * near_hand
        c = L[f"{f}1.L"] + n_h * 0.009 - a_h * 0.006
        extra += H["pads"] * np.exp(-(np.linalg.norm(arm.verts - c, axis=1) / 0.0095) ** 2) * palmar * near_hand
    c = L["wrist.L"] + a_h * 0.032 + r_h * 0.022 + n_h * 0.010
    extra += H["hypothenar"] * np.exp(-(np.linalg.norm(arm.verts - c, axis=1) / 0.021) ** 2) * palmar * near_hand
    arm.verts = arm.verts + out * extra[:, None]
    arm.rings = arm.verts[idx]
    # web quads between neighbouring fingers (outward = distal)
    last = idx[-1]
    for fi in range(3):
        quad = [last[4 * fi + 3], last[4 * fi + 4], last[27 - 4 * fi], last[28 - 4 * fi]]
        v = arm.verts[quad]
        nrm = np.cross(v[1] - v[0], v[3] - v[0])
        if nrm @ t_end < 0:
            quad = quad[::-1]
        arm.faces.append(quad)
        u = ATLAS["arm"]
        arm.uv.append(np.array([[u[2], u[3]], [u[2], u[3]], [u[2], u[3]], [u[2], u[3]]]))
        arm.face_mat.append(0)
    arm.info = dict(kind="arm", path=path, la=la, lf=lf, lp=lp, s_web=s_web, s_mcp=s_web - H["web"], web=H["web"],
                    amap=amap)
    for f in FINGERS:
        fs = fingers[f]
        fs.ext = {j: ("arm_L", int(last[amap[(f, j)]] - 0)) for j in range(8)}
    # arm.ring_index entries are local indices of the arm shell = its own vertex numbers (it has no ext)
    return arm, fingers, thumb


# ---------------------------------------------------------------- leg + foot
def foot_outline(shape):
    """The side silhouette (y, z) of the lower shin, ankle and foot as a closed polygon: Achilles, heel, flat sole at the
    shoe's inner floor, toes, instep, front of the shin. Built from the foot numbers and the landmarks."""
    L = shape.land
    F = shape.dims["foot"]
    A, ball = L["ankle.L"], L["toe.L"]
    yA, yb, zb = float(A[1]), float(ball[1]), float(ball[2])
    zf = float(F["foot_inner_floor_z"])
    yh = float(F["shoe_y"][1]) - 0.008
    yt = yh - float(F["foot_length"])
    tz = zb + 0.0165
    pts = [
        (yA + 0.0240, 0.200), (yA + 0.0232, 0.150), (yA + 0.0245, 0.115), (yA + 0.0290, 0.090), (yA + 0.0350, 0.070),
        (yh - 0.0030, 0.052), (yh, 0.040), (yh - 0.0020, 0.030), (yh - 0.0080, zf + 0.003), (yh - 0.0160, zf),
        (yA - 0.020, zf), (yb + 0.025, zf), (yb, zf), (yt + 0.020, zf + 0.001),
        (yt + 0.007, zf + 0.004), (yt + 0.001, zf + 0.010), (yt, zf + 0.016), (yt + 0.003, zf + 0.022),
        (yt + 0.012, zf + 0.026), (yt + 0.040, zf + 0.034),
        (yb, tz), (yb + 0.028, 0.063), (yA - 0.0300, 0.0745), (yA - 0.0200, 0.0850), (yA - 0.0185, 0.0960),
        (yA - 0.0210, 0.120), (yA - 0.0235, 0.150), (yA - 0.0240, 0.200),
    ]
    return np.array(pts, float)


def _ray_extents(c2, d2, poly):
    """Distances from point c2 along +d2 and -d2 to the closed polygon `poly` (first intersection each way)."""
    best = [np.inf, np.inf]
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        e = q - p
        den = d2[0] * e[1] - d2[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = p - c2
        lam = (w[0] * e[1] - w[1] * e[0]) / den
        mu = (w[0] * d2[1] - w[1] * d2[0]) / den
        if 0.0 <= mu <= 1.0:
            if lam > 1e-9 and lam < best[0]:
                best[0] = lam
            elif lam < -1e-9 and -lam < best[1]:
                best[1] = -lam
    return best


def leg_path(shape):
    """The leg's centre line: hip, knee, down the shin past the ankle, a rounded bend, then forward over the foot."""
    L = shape.land
    F = shape.dims["foot"]
    H, K, A = L["leg.L"], L["knee.L"], L["ankle.L"]
    ball = L["toe.L"]
    zf = float(F["foot_inner_floor_z"])
    heel_y = float(F["shoe_y"][1]) - 0.008
    y_toe = heel_y - float(F["foot_length"])
    R = float(F.get("bend_radius", 0.028))
    z_end = float(F.get("path_z", 0.0475))              # height of the centre line where the bend ends
    corner = np.array([A[0], A[1] + 0.001, z_end + R])
    p_ball = np.array([ball[0], ball[1], 0.5 * (z_end + ball[2])])
    front = np.array([L["toe_end.L"][0], y_toe + 0.016, zf + 0.0215])
    pts = [H, K, corner, p_ball, front]
    dense = G.fillet_polyline(pts, [0, 0, R, 0.05, 0], step=0.003)
    return Path(dense, blend=0.0), pts


def leg_shell(shape, M=24):
    D = shape.dims["limb"]
    F = shape.dims["foot"]
    path, pts = leg_path(shape)
    L = shape.land
    H, K, A = L["leg.L"], L["knee.L"], L["ankle.L"]
    ball, tip = L["toe.L"], L["toe_end.L"]
    sK, sA = path.project(np.array([K, A]))
    sB = path.project(np.array([ball]))[0]
    sEnd = path.length
    th, sh_ = sK, sA - sK
    tt, tm, tl, kn = _half(D["thigh_top"]), _half(D["thigh_mid"]), _half(D["thigh_low"]), _half(D["knee"])
    ca, sl, an = _half(D["calf"]), _half(D["shin_low"]), _half(D["ankle"])
    ca = (ca[0] * 1.035, ca[1] * 1.03)
    an = (an[0] * 0.95, an[1] * 0.93)
    fw = 0.5 * float(F["foot_width"])
    R = float(F.get("bend_radius", 0.028))
    s_arc0 = float(path.project(np.array([[A[0], A[1], pts[2][2] + R]]))[0])
    s_arc1 = s_arc0 + 0.5 * np.pi * R * 1.02
    s_foot0 = s_arc0 - 0.020
    # limb table (thigh, knee, calf, ankle): rx front-back half extent, ry lateral half extent, ox shift to the front,
    # oy shift to the midline; thighs sit behind and inside the hip joint landmark
    rows = [
        (0.0, tt[0], tt[1], -0.012, 0.021, 2.0),
        (0.10 * th, tt[0], tt[1], -0.012, 0.021, 2.0),
        (0.50 * th, tm[0], tm[1], -0.008, 0.009, 2.0),
        (0.85 * th, tl[0], tl[1], -0.002, 0.002, 2.0),
        (th, kn[0], kn[1], 0.0, 0.0, 2.0),
        (sK + 0.25 * sh_, ca[0], ca[1], -0.0055, 0.0, 2.0),
        (sK + 0.43 * sh_, 0.5 * (ca[0] + sl[0]), 0.5 * (ca[1] + sl[1]), -0.0040, 0.0, 2.0),
        (sK + 0.62 * sh_, sl[0], sl[1], -0.002, 0.0, 2.0),
        (sK + 0.85 * sh_, an[0], an[1], -0.001, 0.0, 2.05),
        (s_foot0, an[0], an[1] * 1.04, -0.001, 0.0, 2.1),
        (s_arc0, an[0] * 1.02, fw * 0.66, -0.001, 0.0, 2.2),
        (0.5 * (s_arc0 + s_arc1), 0.026, fw * 0.80, 0.0, 0.0, 2.4),
        (s_arc1, 0.022, fw * 0.92, 0.0, 0.0, 2.6),
        (sB, 0.0165, fw, 0.0, 0.0, 2.7),
        (sB + 0.025, 0.014, fw * 0.95, 0.0, 0.0, 2.7),
        (sB + 0.045, 0.012, fw * 0.80, 0.0, 0.0, 2.6),
        (sEnd - 0.014, 0.010, fw * 0.50, 0.0, 0.0, 2.4),
        (sEnd, 0.008, fw * 0.28, 0.0, 0.0, 2.2),
    ]
    R_ = np.array(rows)
    tab = Table(R_[:, 0], rx=R_[:, 1], ry=R_[:, 2], ox=R_[:, 3], oy=R_[:, 4], n=R_[:, 5])
    ss = [0.0, 0.03, 0.06, 0.10 * th, 0.20 * th, 0.32 * th, 0.45 * th, 0.58 * th, 0.70 * th, 0.82 * th, 0.92 * th, th,
          sK + 0.07 * sh_, sK + 0.16 * sh_, sK + 0.27 * sh_, sK + 0.40 * sh_, sK + 0.54 * sh_, sK + 0.68 * sh_,
          sK + 0.80 * sh_, s_foot0, sA - 0.004, s_arc0]
    ss += list(s_arc0 + np.linspace(0.0, 1.0, 9)[1:] * (s_arc1 - s_arc0))
    ss += [s_arc1 + 0.4 * (sB - s_arc1), sB - 0.012, sB, sB + 0.012, sB + 0.024, sB + 0.036, sB + 0.048,
           sEnd - 0.020, sEnd - 0.010, sEnd]
    ss = sorted(set(round(float(v), 7) for v in ss if 0.0 <= v <= sEnd))
    ss_a = np.array(ss)
    # outline-driven front/back extents for the ankle and foot: the sole stays flat, the heel and instep follow the
    # silhouette whatever the bend does to the ring planes
    t_ = path.tangent(ss_a)
    ef = G.rmf_frames(t_, (0.0, -1.0, 0.0))
    cc = path.point(ss_a)
    poly = foot_outline(shape)
    sec = tab(ss_a)
    rx, ox = np.array(sec["rx"], float), np.array(sec["ox"], float)
    for i, s in enumerate(ss_a):
        w = float(G.smoothstep((s - s_foot0) / max(s_arc0 - s_foot0, 1e-6)))
        if w <= 0.0:
            continue
        d2 = ef[i][1:3] / max(np.linalg.norm(ef[i][1:3]), 1e-9)
        tp, tn = _ray_extents(cc[i][1:3], d2, poly)
        if np.isfinite(tp) and np.isfinite(tn):
            rx[i] = (1 - w) * rx[i] + w * 0.5 * (tp + tn)
            ox[i] = (1 - w) * ox[i] + w * 0.5 * (tp - tn)

    def section(s):
        return dict(rx=rx, ry=np.array(sec["ry"]), ox=ox, oy=np.array(sec["oy"]), n=np.array(sec["n"]))

    sh = tube("leg_L", path, ss_a, section, M, ("rmf", (0.0, -1.0, 0.0)), cap0="dome", cap1="dome", cap_rings=4,
              uv_rect=ATLAS["leg"], cap_len=(1.0, 1.0))
    # kneecap hint and ankle bones: soft outward bumps along the ring normals
    cen = path.point(np.maximum(sh.s, 0.0))
    out = G.unit(sh.verts - cen)
    ring_v = sh.ring >= 0
    bump = np.zeros(len(sh.verts))
    cp = np.array([K[0], K[1] - kn[0], K[2] + 0.010])
    front = (out @ np.array([0.0, -1.0, 0.0])) > 0.25
    bump += 0.0032 * np.exp(-(np.linalg.norm(sh.verts - cp, axis=1) / 0.020) ** 2) * front * ring_v
    for side, (dx, dy, dz, amp, rad) in {"lat": (an[1], 0.004, -0.008, 0.0024, 0.011),
                                         "med": (-an[1], -0.002, 0.003, 0.0020, 0.010)}.items():
        cm = np.array([A[0] + dx, A[1] + dy, A[2] + dz])
        bump += amp * np.exp(-(np.linalg.norm(sh.verts - cm, axis=1) / rad) ** 2) * ring_v
    sh.verts = sh.verts + out * bump[:, None]
    sh.rings = sh.verts[sh.ring_index]
    sh.info = dict(kind="leg", path=path, sK=sK, sA=sA, sC=s_arc0, sB=sB, sEnd=sEnd)
    return sh


def build_shells(shape, M_torso=32, M_arm=32, M_leg=24, nails=True):
    """All shells, left and right: [torso, arm_L, fingers_L..., thumb_L, leg_L, nails_L, then the right side mirrored]."""
    arm, fingers, thumb = arm_hand_shells(shape, M_arm)
    left = [arm] + [fingers[f] for f in FINGERS] + [thumb, leg_shell(shape, M_leg)]
    if nails:
        left += [nail_shell(shape, s) for s in [fingers[f] for f in FINGERS] + [thumb]]
    shells = [torso_shell(shape, M_torso)]
    shells += left
    for sh in left:
        m = mirror_x(sh, sh.name[:-2] + "_R")
        m.info = sh.info
        shells.append(m)
    return shells
