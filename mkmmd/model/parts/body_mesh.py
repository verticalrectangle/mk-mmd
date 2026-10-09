"""Surface of the body part: torso + neck (one loft up to the neck seam ring), an arm (one loft from inside the shoulder
to the wrist seam), the hand (body_hand's designed surface, or the mesh body_hand_mesh places, welded to the arm's last
ring) and a leg with its foot (one bent loft). Left side built, right side mirrored. Sizes come from `Shape.dims` (torso
cuts, limb diameters, foot), `Shape.hand` (the hand's design) and `Shape.land`.

Shell names: torso, arm_L, hand_L, leg_L, nail_<finger>_L (index middle ring little thumb), and the same with _R.
Atlas tiles (u0, v0, u1, v1, v up) place every shell in the skin texture; the hand uses the palm and finger tiles."""
import numpy as np

from . import body_geom as G
from . import body_hand, body_hand_mesh
from .body_geom import Path, Table, tube, mirror_x, pchip

ATLAS = {
    "torso": (0.00, 0.50, 0.50, 1.00),
    "arm": (0.50, 0.50, 0.75, 1.00),
    "leg": (0.75, 0.50, 1.00, 1.00),
    "index": (0.00, 0.25, 0.20, 0.50),
    "middle": (0.20, 0.25, 0.40, 0.50),
    "ring": (0.40, 0.25, 0.60, 0.50),
    "little": (0.60, 0.25, 0.80, 0.50),
    "thumb": (0.80, 0.25, 1.00, 0.50),
    "palm": (0.00, 0.00, 0.50, 0.25),
}


def _seg(a, b):
    return float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))


def _half(pair):
    return 0.5 * float(pair[0]), 0.5 * float(pair[1])


# ---------------------------------------------------------------- torso + neck
TORSO_REF = 0.76                    # the widest cut of the torso the heights below (exponents, bust) were tuned on


def _rise(shape):
    """How far this torso stands above the one the tuned heights were made on: its widest cut against TORSO_REF (0 for
    the base; [proportions] leg_extra raises the torso and these heights with it)."""
    T = shape.dims["torso"]
    return float(np.asarray(T["z"], float)[int(np.argmax(np.asarray(T["width"], float)))]) - TORSO_REF


def torso_rows(shape):
    """(z, half width, front y, back y, n) rows of the torso cuts: the pelvis tapers into the crotch below the hips (the
    cuts include the thighs there), and extra rows refine the shoulder line where it turns into the neck. The last row is
    the neck seam ring (a plain ellipse)."""
    T = shape.dims["torso"]
    z = np.array(T["z"], float)
    hw = np.array(T["width"], float) / 2
    yf = np.array(T["y_front"], float)
    yb = np.array(T["y_back"], float)
    up = _rise(shape)
    nz = pchip([z[0] - 0.08, up + 0.80, up + 0.93, up + 1.03, up + 1.10, up + 1.14, z[-1]],
               [2.4, 2.5, 2.25, 2.3, 2.5, 2.3, 2.0])
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
        sel = (z > up + 0.97) & (z < zt - 0.05)
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
    bust = np.exp(-((z - (_rise(shape) + 1.03)) / 0.045) ** 2) * 0.010
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


# ---------------------------------------------------------------- arm (the hand is welded at the seam)
def arm_shell(shape, M=32):
    """The arm tube from inside the shoulder to the seam `body_hand.SEAM` before the wrist joint, open there: the hand is
    welded onto its last ring, so M is the hand's seam size (hand_module(...).seam_size). Ring point k sits 2 pi k / M
    from the thumb side (forward at rest) towards the palm side. The path runs on past the wrist along the hand, and the
    section table to just beyond the wrist, so the hand can sample the forearm's surface there."""
    L = shape.land
    D = shape.dims["limb"]
    J, E, W = L["arm.L"], L["elbow.L"], L["wrist.L"]
    path = Path([J, E, W, W + shape.frame[0] * 0.04], blend=0.035)
    la, lf = _seg(J, E), _seg(E, W)
    s_w = la + lf
    s_seam = s_w + body_hand.SEAM
    top, mid, el = _half(D["upper_arm_top"]), _half(D["upper_arm_mid"]), _half(D["elbow"])
    ft, fm, wr = _half(D["forearm_top"]), _half(D["forearm_mid"]), _half(D["wrist"])
    ball = 1.30
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
        (s_w + 0.02, wr[0], wr[1], 0.0, 0.0, 2.6),
    ]
    R = np.array(rows)
    tab = Table(R[:, 0], rx=R[:, 1], ry=R[:, 2], ox=R[:, 3], oy=R[:, 4], n=R[:, 5])
    ss = [f * la for f in (0.0, 0.08, 0.18, 0.32, 0.46, 0.60, 0.74, 0.86, 0.93, 1.0)]
    ss += [la + f * lf for f in (0.07, 0.16, 0.30, 0.45, 0.60, 0.74, 0.85)] + [s_seam]
    ref = (0.0, -1.0, 0.0)
    arm = tube("arm_L", path, ss, tab, M, ref, cap0="dome", cap1="open", cap_rings=3, theta0=0.0,
               uv_rect=ATLAS["arm"], cap_len=(1.0, 1.0))
    arm.info = dict(kind="arm", path=path, tab=tab, ref=ref, la=la, lf=lf, s_w=s_w, s_seam=s_seam, M=M)
    return arm


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


def hand_module(D):
    """The module that makes the hand of design D: body_hand_mesh for a `mesh`, else body_hand."""
    return body_hand_mesh if D["mesh"] else body_hand


def build_shells(shape, M_torso=32, M_arm=None, M_leg=24, nails=True):
    """All shells, left and right: [torso, arm_L, hand_L, leg_L, nails_L..., then the right side mirrored]. The hand is
    welded onto the arm's last ring, which has as many points as the hand's seam (M_arm, when given, must say so)."""
    hm = hand_module(shape.hand)
    seam = hm.seam_size(shape.hand)
    if M_arm is not None and int(M_arm) != seam:
        raise ValueError(f"[body.resolution] arm must be {seam} (the hand's seam ring), got {M_arm}")
    arm = arm_shell(shape, seam)
    hand, nail_shells = hm.hand_shells(shape, arm, ATLAS, nails=nails)
    left = [arm, hand, leg_shell(shape, M_leg)] + nail_shells
    shells = [torso_shell(shape, M_torso)]
    shells += left
    for sh in left:
        m = mirror_x(sh, sh.name[:-2] + "_R")
        m.info = sh.info
        shells.append(m)
    return shells
