"""Stand-in head and body parts for the hair builders' tests and previews (numpy only).

They publish the same `info` keys as the real head and body parts (see mkmmd/model/parts/hair_fit.py), with simple
shapes built from the numbers of the proportions study (an egg-shaped head 0.19 m wide and 0.2355 m tall on a 1.4267 m
body, eyes low in the face, big hair volume left to the hair part). Use `make_ctx(tmp_path, spec)` to get a real
`BuildCtx` with `parts` filled in, then call the hair builder."""
import numpy as np

from mkmmd.model.build import BuildCtx
from mkmmd.model.part import Bone, Mesh, Part, RigidBody
from mkmmd.model.parts.hair_fit import dirs, unit

C = np.array([0.0, -0.0148, 1.27])                      # head centre used by the stand-in (ear-hole height ~1.25)
CRANIUM_C = np.array([0.0, -0.0148, 1.30])
CRANIUM_R = np.array([0.0948, 0.0987, 0.0988])          # width 0.1895, depth 0.1974, top 1.3988

# outline of the face shell (RinStudy): z, full width, y of the front
OZ = np.array([1.1672, 1.1770, 1.1869, 1.1967, 1.2066, 1.2164, 1.2263, 1.2361, 1.2460, 1.2558, 1.2657, 1.2755, 1.2854,
               1.2952, 1.3051, 1.3149, 1.3248, 1.3346, 1.3445, 1.3543, 1.3642, 1.3740, 1.3839, 1.3937])
OW = np.array([0.0462, 0.0458, 0.0588, 0.0954, 0.1198, 0.1335, 0.1421, 0.1493, 0.1562, 0.1626, 0.1683, 0.1739, 0.1790,
               0.1832, 0.1864, 0.1884, 0.1895, 0.1889, 0.1849, 0.1790, 0.1698, 0.1541, 0.1308, 0.0898])
OY = np.array([-0.0263, -0.0282, -0.0913, -0.0973, -0.1026, -0.1107, -0.1167, -0.1144, -0.1060, -0.1040, -0.1041,
               -0.1052, -0.1074, -0.1101, -0.1121, -0.1131, -0.1131, -0.1123, -0.1093, -0.1057, -0.1003, -0.0949,
               -0.0885, -0.0821])
NOSE_Z = (1.2164, 1.2460)


def _front_y(z):
    z = np.asarray(z, float)
    keep = (OZ < NOSE_Z[0]) | (OZ > NOSE_Z[1])               # the nose is not part of the stand-in
    return np.interp(z, OZ[keep], OY[keep])


def _half_w(z):
    z = np.asarray(z, float)
    cran = CRANIUM_R[0] * np.sqrt(np.clip(1 - ((z - CRANIUM_C[2]) / CRANIUM_R[2]) ** 2, 0, 1))
    return np.maximum(np.where(z < OZ[-1], np.interp(z, OZ, OW) / 2, 0.0), cran)


def _back_y(z):
    z = np.asarray(z, float)
    low = 0.0359 + (0.0839 - 0.0359) * np.clip((z - 1.17) / 0.13, 0, 1) ** 1.5 * (3 - 2 * np.clip((z - 1.17) / 0.13, 0, 1))
    cran = CRANIUM_C[1] + CRANIUM_R[1] * np.sqrt(np.clip(1 - ((z - CRANIUM_C[2]) / CRANIUM_R[2]) ** 2, 0, 1))
    return np.where(z <= CRANIUM_C[2], low, cran)


def _ring(z, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    s = np.sin(a)
    hw = float(_half_w(z))
    yf, yb = float(_front_y(z)), float(_back_y(z))
    return np.stack([hw * np.cos(a), np.where(s > 0, yb * s, -yf * s), np.full(n, z)], -1)


def _loft(rings, cap_bottom=True, cap_top=True):
    n = len(rings[0])
    verts = np.concatenate(rings, 0)
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            faces.append([a, b, b + n, a + n])
    if cap_bottom:
        verts = np.vstack([verts, rings[0].mean(0)])
        c = len(verts) - 1
        faces += [[c, (j + 1) % n, j] for j in range(n)]
    if cap_top:
        verts = np.vstack([verts, rings[-1].mean(0)])
        c = len(verts) - 1
        k = (len(rings) - 1) * n
        faces += [[c, k + j, k + (j + 1) % n] for j in range(n)]
    return verts, faces


def skin_mesh():
    """A closed head + neck shell: loft of rings from the neck (z 1.10) up over the skull to z 1.3988."""
    zs = np.concatenate([np.linspace(1.10, 1.1672, 4)[:-1], OZ[:-1], np.linspace(OZ[-1], 1.3980, 6)])
    rings = []
    for z in zs:
        if z < OZ[0]:                                           # neck: half width 0.0229, front 0.0246, back 0.0359
            a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            s = np.sin(a)
            rings.append(np.stack([0.0229 * np.cos(a), np.where(s > 0, 0.0359 * s, 0.0246 * s), np.full(48, z)], -1))
        else:
            rings.append(_ring(z))
    # y of the neck rings is measured from y = 0 like the others (front -0.0246, back +0.0359)
    v, f = _loft(rings)
    return v, f


def surf_point(x, z, side="front"):
    """Point on the front surface of the stand-in head at (x, z)."""
    hw = float(_half_w(z))
    yf = float(_front_y(z))
    return np.array([x, -abs(yf) * np.sqrt(max(1 - (x / hw) ** 2, 0.0)), z])


def head_info():
    """The head part's info with stand-in values."""
    verts, faces = skin_mesh()
    top = np.array([0.0, -0.0148, 1.3988])
    # hairline arch from the left temple over the forehead to the right temple
    hair = []
    for x in np.linspace(0.082, -0.082, 29):                      # the temple ends well in front of the ears
        z = 1.3869 - (1.3869 - 1.3297) * (x / 0.082) ** 2
        hair.append(surf_point(x, z))
    hair = np.array(hair)
    ears, side, cat, eyes, brows = {}, {}, {}, {}, {}
    outline_parts = [hair]
    for s, sg in (("L", 1.0), ("R", -1.0)):
        zs = np.linspace(1.3297, 1.2060, 8)
        side[s] = np.array([[sg * np.sqrt(max(1 - (0.052 / float(_front_y(z))) ** 2, 0)) * float(_half_w(z)), -0.052, z]
                            for z in zs])
        ears[s] = {"top": np.array([sg * 0.090, -0.001, 1.268]), "lobe": np.array([sg * 0.077, -0.006, 1.233]),
                   "front": np.array([sg * 0.085, -0.013, 1.248]), "back": np.array([sg * 0.082, 0.010, 1.253]),
                   "centre": np.array([sg * 0.081, -0.002, 1.250]), "out": unit(np.array([sg, 0.1, 0.0]))}
        o = np.array([sg * 0.0709, -0.0294, 1.3638])
        o = _project_to_skin(o, verts, faces)                      # RinStudy's guide point, snapped to the skin
        n = _skin_normal(o)
        up = unit(np.array([0, 0, 1.0]) - n * n[2])
        out = unit(np.array([sg, 0.0, 0.0]) - n * (sg * n[0]))
        cat[s] = {"origin": o, "normal": n, "up": up, "out": out}
        ex = sg * 0.0476
        eyes[s] = {"center": np.array([ex, -0.0691, 1.2604]), "pupil": np.array([sg * 0.0422, -0.0863, 1.2604]),
                   "radius": 0.0225, "inner": np.array([sg * 0.0157, -0.100, 1.2590]),
                   "outer": np.array([sg * 0.0842, -0.083, 1.2640]),
                   "lid_top": np.array([sg * 0.0422, -0.0920, 1.2852]), "lid_bottom": np.array([sg * 0.0422, -0.090, 1.2356])}
        bx = sg * np.linspace(0.0256, 0.0768, 8)
        bz = 1.3036 + 0.004 * np.sin(np.linspace(0, np.pi, 8))
        brows[s] = np.array([surf_point(x, z) + np.array([0, -0.001, 0]) for x, z in zip(bx, bz)])
    nape = []
    for ang in np.linspace(20, 160, 29):                         # back half of the ring at z = 1.19 (+X side first)
        a = np.radians(ang)
        z = 1.19
        nape.append([float(_half_w(z)) * np.cos(a), float(_back_y(z)) * np.sin(a), z])
    nape = np.array(nape)
    nape = nape[::1]
    outline = np.concatenate([hair, side["R"][::-1], np.array([[0.0, -0.0811, 1.1778]]), side["L"]])
    ring = np.array([[0.0229 * np.cos(a), (0.0359 if np.sin(a) > 0 else -0.0246) * np.sin(a), 1.159] for a in
                     np.linspace(0, 2 * np.pi, 16, endpoint=False)])
    return {"head_center": C.copy(), "head_radii": tuple(CRANIUM_R), "skull_top": top,
            "brow_front": surf_point(0.0, 1.3036), "chin": np.array([0.0, -0.0811, 1.1778]),
            "skin": {"verts": verts, "faces": faces}, "hairline": hair, "hairline_side": side, "nape": nape,
            "face_outline": outline, "ears": ears, "cat_ear_anchors": cat, "eyes": eyes, "brows": brows,
            "neck_ring": ring, "neck_center": np.array([0.0, 0.005, 1.159])}


_SKIN_CACHE = {}


def _skin_ray(p, d):
    from mkmmd.model.parts.hair_fit import raycast_first
    key = "tris"
    if key not in _SKIN_CACHE:
        v, f = skin_mesh()
        _SKIN_CACHE[key] = (v, np.array([[x[0], x[k], x[k + 1]] for x in f for k in range(1, len(x) - 1)]))
    v, t = _SKIN_CACHE[key]
    return raycast_first(p, np.asarray([d], float), v, t)[0]


def _project_to_skin(p, verts=None, faces=None):
    d = unit(p - C)
    t = _skin_ray(C, d)
    return C + d * t


def _skin_normal(p):
    d = unit(p - C)
    e1 = unit(np.cross([0, 0, 1.0], d))
    e2 = unit(np.cross(d, e1))
    pts = [C + unit(d + eps * e) * _skin_ray(C, unit(d + eps * e)) for eps, e in ((0, e1 * 0), (0.02, e1), (0.02, e2))]
    n = unit(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
    return n if n @ d > 0 else -n


# ---------------------------------------------------------------- the mannequin body
TZ = [0.7220, 0.7410, 0.7600, 0.7790, 0.7980, 0.8170, 0.8360, 0.8550, 0.8740, 0.8930, 0.9120, 0.9310, 0.9500, 0.9690,
      0.9880, 1.0070, 1.0260, 1.0450, 1.0640, 1.0830, 1.1020, 1.1210, 1.1400, 1.1590]
TW = [0.2110, 0.2336, 0.2458, 0.2408, 0.2322, 0.2215, 0.2027, 0.1864, 0.1708, 0.1555, 0.1426, 0.1337, 0.1359, 0.1412,
      0.1506, 0.1570, 0.1687, 0.1730, 0.1703, 0.1675, 0.1436, 0.1060, 0.0763, 0.0458]
TF = [-0.0805, -0.1012, -0.1064, -0.1081, -0.1088, -0.1063, -0.1059, -0.1071, -0.1073, -0.1054, -0.1050, -0.1045,
      -0.1043, -0.1036, -0.1017, -0.1025, -0.1095, -0.0866, -0.0825, -0.0724, -0.0558, -0.0406, -0.0238, -0.0246]
TB = [0.0462, 0.0547, 0.0584, 0.0587, 0.0564, 0.0495, 0.0390, 0.0243, 0.0154, 0.0080, 0.0056, 0.0066, 0.0141, 0.0218,
      0.0333, 0.0404, 0.0452, 0.0499, 0.0536, 0.0565, 0.0562, 0.0535, 0.0456, 0.0359]

LM = {"root": [0.0, -0.0027, -0.0036], "center": [0.0, -0.0167, 0.8419], "groove": [0.0, -0.0167, 0.8564],
      "waist": [0.0, -0.0508, 0.8823], "lower_body": [0.0, -0.0508, 0.8766], "upper_body": [0.0, -0.0508, 0.8880],
      "upper_body2": [0.0, -0.0479, 0.9680], "neck": [0.0, 0.0020, 1.1563], "head": [0.0, -0.0045, 1.2041],
      "head_tip": [0.0001, -0.0133, 1.3483], "shoulder.L": [0.0438, 0.0084, 1.1265], "arm.L": [0.0868, 0.0057, 1.1206],
      "elbow.L": [0.2539, 0.0132, 1.0140], "wrist.L": [0.3941, 0.0061, 0.9418], "leg.L": [0.0755, -0.0356, 0.7425],
      "knee.L": [0.0527, -0.0320, 0.4189], "ankle.L": [0.0602, -0.0008, 0.0936]}


def _tube(a, b, ra, rb, n=14, rings=5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = unit(b - a)
    u = unit(np.cross(ax, [0.0, 1.0, 0.0] if abs(ax[1]) < 0.9 else [1.0, 0.0, 0.0]))
    v = np.cross(ax, u)
    out = []
    for t in np.linspace(0, 1, rings):
        c = a + (b - a) * t
        r = ra + (rb - ra) * t
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        out.append(c + r * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)))
    return _loft(out)


def _torso_ring(z, hw, yf, yb, n=24):
    """Elliptic ring: half width hw, front depth yf (>0), back depth yb (>0), y measured from 0."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    s = np.sin(a)
    return np.stack([hw * np.cos(a), np.where(s > 0, yb * s, yf * s), np.full(n, z)], -1)


def body_part():
    rings = [_torso_ring(z, w / 2, abs(f), b) for z, w, f, b in zip(TZ, TW, TF, TB)]
    V, F = _loft(rings)
    parts = [(V, F)]
    for sg in (1.0, -1.0):
        m = lambda p: [sg * p[0], p[1], p[2]]
        parts.append(_tube(m(LM["arm.L"]), m(LM["elbow.L"]), 0.031, 0.026))
        parts.append(_tube(m(LM["elbow.L"]), m(LM["wrist.L"]), 0.026, 0.021))
        parts.append(_tube(m(LM["leg.L"]), m(LM["knee.L"]), 0.060, 0.040))
        parts.append(_tube(m(LM["knee.L"]), m(LM["ankle.L"]), 0.040, 0.025))
    off, VV, FF = 0, [], []
    for v, f in parts:
        VV.append(v)
        FF += [[i + off for i in fc] for fc in f]
        off += len(v)
    VV = np.concatenate(VV, 0)
    bone = lambda n, h, p="": Bone(n, tuple(h), parent=p)
    bones = [bone("全ての親", (0, 0, 0)), bone("センター", LM["center"], "全ての親"), bone("下半身", LM["lower_body"], "センター"),
             bone("上半身", LM["upper_body"], "センター"), bone("上半身2", LM["upper_body2"], "上半身"),
             bone("首", LM["neck"], "上半身2"), bone("頭", LM["head"], "首")]
    from mkmmd.model.parts.hair_rig import euler_for_axis

    def cap(name, bone_name, a, b, r):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return RigidBody(name, bone_name, "capsule", (r, max(float(np.linalg.norm(b - a)) - 2 * r, 0.001), 0.0),
                         tuple((a + b) / 2), euler_for_axis(b - a), "static", 0, (0,), 1.0)
    bodies = [RigidBody("col_head", "頭", "sphere", (0.094, 0, 0), (0.0, -0.0148, 1.3048), (0, 0, 0), "static", 0, (0,), 1.0),
              cap("col_neck", "首", (0, 0.002, 1.1563), (0, -0.0045, 1.2041), 0.0266),
              cap("col_upper_body2", "上半身2", (0, -0.034, 0.978), (0, -0.030, 1.1263), 0.074),
              cap("col_upper_body", "上半身", (0, -0.049, 0.9066), (0, -0.045, 0.968), 0.058),
              cap("col_lower_body", "下半身", (-0.075, -0.024, 0.79), (0.075, -0.024, 0.79), 0.082)]
    for s, sg in (("L", 1.0), ("R", -1.0)):
        m = lambda p: [sg * p[0], p[1], p[2]]
        bodies += [cap(f"col_shoulder_{s}", "左肩" if sg > 0 else "右肩", m(LM["shoulder.L"]), m(LM["arm.L"]), 0.034),
                   cap(f"col_arm_{s}", "左腕" if sg > 0 else "右腕", m(LM["arm.L"]), m(LM["elbow.L"]), 0.0266),
                   cap(f"col_thigh_{s}", "左足" if sg > 0 else "右足", m(LM["leg.L"]), m(LM["knee.L"]), 0.0478)]
        bones += [bone("左肩" if sg > 0 else "右肩", m(LM["shoulder.L"]), "上半身2"),
                  bone("左腕" if sg > 0 else "右腕", m(LM["arm.L"]), "左肩" if sg > 0 else "右肩"),
                  bone("左足" if sg > 0 else "右足", m(LM["leg.L"]), "下半身")]
    landmarks = {k: np.array(v) for k, v in LM.items()}
    for k in [k for k in LM if k.endswith(".L")]:
        landmarks[k[:-2] + ".R"] = np.array([-LM[k][0], LM[k][1], LM[k][2]])
    landmarks["tail_root"] = np.array([0.0, 0.0095, 0.90])
    info = {"landmarks": landmarks, "tail_normal": unit(np.array([0.0, 1.0, 0.25])),
            "neck_seam": {"center": np.array([0, 0.003, 1.159]), "n": 16},
            "torso_profile": {"rows": TZ, "y_front": TF, "y_back": TB, "half_width": [w / 2 for w in TW]}}
    return Part("body", meshes=[Mesh("body", VV, FF, None)], bones=bones, bodies=bodies, info=info)


def head_part():
    return Part("head", info=head_info())


def make_ctx(tmp_path, spec=None, seed=1):
    """A real BuildCtx with the stand-in body and head built and the part name set to "hair"."""
    spec = dict(spec or {})
    spec.setdefault("model", {"name": "test", "parts": ["body", "head", "hair"], "seed": seed})
    ctx = BuildCtx(spec, tmp_path / "tex", seed=seed)
    body, head = body_part(), head_part()
    ctx.parts["body"], ctx.parts["head"] = body, head
    for k, v in body.info["landmarks"].items():
        ctx.land[k] = np.asarray(v, float)
    ctx.part = "hair"
    return ctx
