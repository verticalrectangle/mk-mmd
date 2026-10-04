"""A simple but complete test character: three parts that prove the whole `mk model` route before real parts exist.

  mannequin        the standard skeleton (all MMD standard bones, IK, twist, fingers), a capsule-and-loft body with
                   analytic weights (incl. twist bones, fingers, toes), a head with eyes (両目/左目/右目), a mouth and
                   the morphs まばたき ウィンク ウィンク右 あ 笑い 照れ, and static rigid bodies hugging the surfaces
  mannequin_hair   a 3-bone 前髪 chain (sheet mesh, dynamic bodies, joints to the head body)
  mannequin_skirt  two 2-bone スカート chains (front, back), a revolved skirt, dynamic bodies and joints

    [model]
    parts = ["mannequin", "mannequin_hair", "mannequin_skirt"]       # any subset: hair and skirt need the body

Textures come from `mkmmd.model.tex` (skin gradient, eye, toon ramps, hair sphere map). Not a character to look at: it is
the fixture for the framework, the assembler, the verification and the pipeline stages."""
import math

import numpy as np

from .. import geo, skin, tex
from .. import skeleton as SK
from ..build import builder
from ..part import Bone, Joint, Material, Morph, Part, RigidBody

HEIGHT = 1.5
HEAD_C = np.array([0.0, -0.005, 1.35])
HEAD_R = np.array([0.083, 0.095, 0.105])
SKIN = (1.0, 0.86, 0.78, 1.0)


def _v(*a):
    return np.array(a, float)


def landmarks(cfg=None):
    """Semantic landmarks of the mannequin (left side + centre)."""
    ang = math.radians(33.0)
    d = _v(math.cos(ang), 0.0, -math.sin(ang))
    arm = _v(0.10, 0.0, 1.10)
    elbow = arm + d * 0.20
    wrist = elbow + d * 0.16
    L = {"upper_body": _v(0, 0, 0.90), "neck": _v(0, 0, 1.16), "head": _v(0, 0, 1.20), "head_tip": _v(0, 0, HEIGHT),
         "shoulder.L": _v(0.04, 0, 1.12), "arm.L": arm, "elbow.L": elbow, "wrist.L": wrist,
         "leg.L": _v(0.075, 0, 0.74), "knee.L": _v(0.07, -0.012, 0.42), "ankle.L": _v(0.07, 0.0, 0.09),
         "toe.L": _v(0.075, -0.13, 0.03), "toe_end.L": _v(0.075, -0.16, 0.025)}
    side = np.cross(d, _v(0, 0, 1))                                  # across the hand (points to -y for the left hand)
    side = side / np.linalg.norm(side)
    palm = wrist + d * 0.075
    layout = {"thumb": (-0.026, (0.0, 0.032, 0.056, 0.074), 0.2), "index": (-0.018, (0.0, 0.028, 0.048, 0.064), 0.0),
              "middle": (0.0, (0.0, 0.032, 0.054, 0.072), 0.0), "ring": (0.017, (0.0, 0.029, 0.049, 0.065), 0.0),
              "little": (0.032, (0.0, 0.024, 0.040, 0.053), 0.0)}
    for fing, (off, along, drop) in layout.items():
        base = palm + side * (-off) * 1.0 + _v(0, 0, 0)
        if fing == "thumb":
            base = wrist + d * 0.02 + _v(0, -0.028, 0.0)
        names = SK.finger_joint_names(fing)
        for n, a in zip(names, along):
            L[f"{n}.L"] = base + d * a + _v(0, 0, -drop * a)
    L["eye.L"] = _v(0.03, -0.072, 1.36)
    return L


def _skin_texture(ctx):
    img = tex.linear_gradient(64, 64, [(0.0, "#ffe2cf"), (1.0, "#f4c7ad")])
    return ctx.save_png("skin", img)


def _toon(ctx, name, shadow):
    return ctx.save_png(name, tex.toon_ramp(shadow, threshold=0.5, softness=0.06))


def _eye_texture(ctx):
    cv = tex.Canvas(96, 96, bg="#00000000")
    cv.ellipse((0.5, 0.5), (0.48, 0.34), fill="#fbf7f2")
    cv.circle((0.5, 0.5), 0.26, fill="#7a1330")
    cv.circle((0.5, 0.5), 0.12, fill="#1c0710")
    cv.circle((0.58, 0.6), 0.06, fill="#ffffff")
    return ctx.save_png("eye", cv.image())


def _mouth_texture(ctx):
    return ctx.save_png("mouth", tex.new(8, 8, "#8f2a3a"))


def _hair_sphere(ctx):
    return ctx.save_png("hair_sphere", tex.sphere_highlight(size=128, band=(0.35, 0.14), strength=0.6, mode="add"))


def _hair_texture(ctx):
    return ctx.save_png("hair", tex.linear_gradient(32, 32, [(0.0, "#7a1d1d"), (1.0, "#d94a3a")]))


def _ring(c, rx, ry, z, n=18):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([c[0] + rx * np.cos(t), c[1] + ry * np.sin(t), np.full(n, z)], axis=1)


def _tube_weights(v, bones, joints, blend):
    return skin.chain_weights(v, bones, joints, blend=blend)


def _to_mesh(g, name, mat, weights, fallback, **kw):
    n = len(g.verts)
    w = skin.normalise(weights, n, fallback=fallback)
    return g.to_mesh(name, mats=[mat], weights=w, **kw)


def _surface_point(x, z):
    """Point on the head ellipsoid above (x, z) on the front (-Y) side."""
    x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
    rel = 1.0 - ((x - HEAD_C[0]) / HEAD_R[0]) ** 2 - ((z - HEAD_C[2]) / HEAD_R[2]) ** 2
    y = HEAD_C[1] - HEAD_R[1] * np.sqrt(np.clip(rel, 0.0, None))
    return np.stack([x, y, z], axis=-1)


def _eye_patch(cx, cz, half_w=0.017, half_h=0.013, n=8, m=8, lift=0.0007):
    def fn(u, v):
        x = cx + (u - 0.5) * 2 * half_w
        z = cz + (v - 0.5) * 2 * half_h
        p = _surface_point(x, z)
        p[..., 1] -= lift
        return p
    return geo.surface(fn, n, m)


@builder("mannequin", needs=())
def build(ctx):
    cfg = ctx.cfg
    L = landmarks(cfg)
    bones = SK.standard_bones(L, {"twist": cfg.get("twist", "standard")})
    C = SK.complete(L, {"twist": cfg.get("twist", "standard")})
    mats = [Material("skin", diffuse=SKIN, ambient=(0.55, 0.45, 0.42), texture=_skin_texture(ctx),
                     toon=_toon(ctx, "toon_skin", "#f0b9a6"), specular=(0.05, 0.05, 0.05), edge_size=0.6),
            Material("shirt", diffuse=(0.35, 0.55, 0.75, 1.0), ambient=(0.3, 0.4, 0.5),
                     toon=_toon(ctx, "toon_shirt", "#9aa7c8")),
            Material("eye", texture=_eye_texture(ctx), alpha_blend=True, edge=False, diffuse=(1, 1, 1, 1)),
            Material("mouth", texture=_mouth_texture(ctx), edge=False, diffuse=(0.9, 0.5, 0.55, 1.0))]
    meshes = []
    sides = (("L", "左", 1.0), ("R", "右", -1.0))

    # ---- torso (loft of ellipses) and neck
    prof = [(0.72, 0.125, 0.085), (0.78, 0.13, 0.09), (0.85, 0.11, 0.08), (0.93, 0.095, 0.07), (1.02, 0.105, 0.075),
            (1.10, 0.125, 0.08), (1.14, 0.08, 0.06), (1.17, 0.05, 0.05)]
    rings = np.stack([_ring((0.0, 0.0), rx, ry, z) for z, rx, ry in prof])
    torso = geo.loft(rings, closed=True, cap_start="fan", cap_end="fan")
    w = skin.chain_weights(torso.verts, ["下半身", "上半身", "上半身2", "首"],
                           [_v(0, 0, 0.70), C["lower_body"], C["upper_body2"], C["neck"], _v(0, 0, 1.24)], blend=0.07)
    meshes.append(_to_mesh(torso, "torso", "shirt", w, "上半身"))
    neck = geo.tube(np.array([_v(0, 0, 1.15), _v(0, 0, 1.19), _v(0, 0, 1.25)]), 0.04, sides=12, cap_start="fan")
    w = skin.chain_weights(neck.verts, ["首", "頭"], [_v(0, 0, 1.12), _v(0, 0, 1.19), _v(0, 0, 1.28)], blend=0.04)
    meshes.append(_to_mesh(neck, "neck", "skin", w, "首"))

    # ---- arms, hands, legs, feet (both sides)
    for sd, jp, sx in sides:
        def P(name, sd=sd):
            return C[f"{name}.{sd}"]
        a, e, wr = P("arm"), P("elbow"), P("wrist")
        path = np.concatenate([np.linspace(a, e, 6)[:-1], np.linspace(e, wr, 6)])
        rad = np.concatenate([np.linspace(0.043, 0.032, 6)[:-1], np.linspace(0.032, 0.024, 6)])
        arm = geo.tube(path, rad, sides=10, cap_start="fan", cap_end="fan")
        joints = [a, P("arm_twist"), e, P("wrist_twist"), wr, wr + (wr - e) * 0.3]
        w = _tube_weights(arm.verts, [f"{jp}腕", f"{jp}腕捩", f"{jp}ひじ", f"{jp}手捩", f"{jp}手首"], joints, 0.05)
        w = skin.mix(w, skin.rigid(len(arm.verts), f"{jp}肩"), skin.sphere_ramp(arm.verts, a, 0.03, 0.06) * 0.5)
        meshes.append(_to_mesh(arm, f"arm_{sd}", "skin", w, f"{jp}腕"))
        # hand: a flat palm and five finger tubes
        d = (wr - e) / np.linalg.norm(wr - e)
        frame = geo.frame_from(x=d, up=_v(0, 0, 1))
        palm = geo.uv_sphere(1.0, center=(0, 0, 0), segments=12, rings=6, scale=(0.045, 0.036, 0.012))
        palm = palm.transformed(R=frame, t=wr + d * 0.05)
        parts_h = [palm]
        wts = [skin.rigid(len(palm.verts), f"{jp}手首")]
        for fing, fjp in (("thumb", "親指"), ("index", "人指"), ("middle", "中指"), ("ring", "薬指"), ("little", "小指")):
            names = SK.finger_joint_names(fing)
            pts = np.array([P(n) for n in names])
            r = np.array([0.0095, 0.0085, 0.0075, 0.006])
            f = geo.tube(pts, r, sides=6, cap_end="fan")
            first = 0 if fing == "thumb" else 1
            fb = [f"{jp}{fjp}{SK._jp_num(first + i)}" for i in range(3)]
            fw = skin.chain_weights(f.verts, fb, pts, blend=0.012)
            parts_h.append(f)
            wts.append(fw)
        hand = geo.merge(parts_h)
        hw = {}
        off = 0
        for part_g, pw in zip(parts_h, wts):
            for bn, arr in pw.items():
                full = hw.setdefault(bn, np.zeros(len(hand.verts)))
                full[off:off + len(part_g.verts)] = arr
            off += len(part_g.verts)
        meshes.append(_to_mesh(hand, f"hand_{sd}", "skin", hw, f"{jp}手首"))
        # leg
        lg, kn, an = P("leg"), P("knee"), P("ankle")
        path = np.concatenate([np.linspace(lg, kn, 6)[:-1], np.linspace(kn, an, 6)])
        rad = np.concatenate([np.linspace(0.065, 0.048, 6)[:-1], np.linspace(0.048, 0.034, 6)])
        leg = geo.tube(path, rad, sides=12, cap_start="fan", cap_end="fan")
        w = skin.chain_weights(leg.verts, [f"{jp}足", f"{jp}ひざ", f"{jp}足首"], [lg, kn, an, an + _v(0, 0, -0.05)],
                               blend=0.07)
        meshes.append(_to_mesh(leg, f"leg_{sd}", "skin", w, f"{jp}足"))
        # foot
        toe, tend = P("toe"), P("toe_end")
        ctr = (an + toe) / 2 + _v(0, -0.01, -0.035)
        foot = geo.uv_sphere(1.0, center=ctr, segments=14, rings=7, scale=(0.036, 0.095, 0.035))
        w = skin.chain_weights(foot.verts, [f"{jp}足首", f"{jp}つま先"], [an, toe, tend], blend=0.04)
        meshes.append(_to_mesh(foot, f"foot_{sd}", "skin", w, f"{jp}足首"))

    # ---- head: ellipsoid, eyes, mouth
    # the head is a coarse cube-sphere cage subdivided once by the assembler (exercises UV, weight and morph carry)
    head = geo.quad_sphere(1.0, center=tuple(HEAD_C), n=4, scale=tuple(HEAD_R * 1.012), uv="spherical")
    w = skin.rigid(len(head.verts), "頭")
    meshes.append(_to_mesh(head, "head", "skin", w, "頭", subsurf=1))
    eye_meshes = []
    for sd, jp, sx in sides:
        e = C["eye.L"] * [sx, 1, 1]
        eye_bone = f"{jp}目"                       # 両目/左目/右目 come from the skeleton (the eye.L landmark)
        patch = _eye_patch(e[0], e[2])
        n = len(patch.verts)
        h = (patch.verts[:, 2] - e[2]) / 0.013
        blink = np.zeros((n, 3))
        blink[:, 2] = -(h + 1.0) * 0.5 * 0.0245
        blink[:, 1] = 0.0004
        wink = f"ウィンク{'' if sd == 'L' else '右'}"
        m = patch.to_mesh(f"eye_{sd}", mats=["eye"], weights={eye_bone: np.ones(n)})
        m.morphs = {"まばたき": blink, wink: blink.copy()}
        eye_meshes.append(m)
    meshes += eye_meshes
    mouth = geo.surface(lambda u, v: _surface_point(-0.022 + u * 0.044, 1.306 + (v - 0.5) * 0.011) - _v(0, 0.0007, 0),
                        10, 4)
    n = len(mouth.verts)
    mm = mouth.to_mesh("mouth", mats=["mouth"], weights={"頭": np.ones(n)})
    vz = (mouth.verts[:, 2] - 1.306) / 0.0055
    open_a = np.zeros((n, 3))
    open_a[:, 2] = -np.clip(0.5 - vz * 0.5, 0, 1) * 0.014
    open_a[:, 1] = np.clip(0.5 - vz * 0.5, 0, 1) * 0.004
    smile = np.zeros((n, 3))
    smile[:, 2] = 0.004 * (np.abs(mouth.verts[:, 0]) / 0.022) ** 2
    mm.morphs = {"あ": open_a, "笑い": smile}
    meshes.append(mm)
    # a blush morph on the head (moves only the cheek vertices)
    hv = head.verts
    cheek = np.exp(-(((np.abs(hv[:, 0]) - 0.05) / 0.025) ** 2 + ((hv[:, 2] - 1.31) / 0.02) ** 2)) * (hv[:, 1] < -0.04)
    blush = np.zeros((len(hv), 3))
    blush[:, 1] = -0.003 * cheek
    next(m_ for m_ in meshes if m_.name == "head").morphs = {"照れ": blush}

    morph_decl = [Morph("まばたき", "eye", "Blink"), Morph("ウィンク", "eye", "Wink"),
                  Morph("ウィンク右", "eye", "Wink R"), Morph("あ", "mouth", "a"), Morph("笑い", "mouth", "Smile"),
                  Morph("照れ", "other", "Blush")]

    # ---- static rigid bodies (head, neck, torso, shoulders, arms, hands, legs, feet)
    rb = []

    def cap(name, bone, a, b, r, **kw):
        rb.append(RigidBody(name, bone, shape="capsule", **geo.capsule_between(a, b, r), mode="static", **kw))
    rb.append(RigidBody("頭", "頭", shape="sphere", size=(0.092, 0, 0), location=tuple(HEAD_C), mode="static", group=0))
    cap("首", "首", _v(0, 0, 1.15), _v(0, 0, 1.2), 0.04)
    cap("上半身", "上半身", C["upper_body"] + _v(0, 0, 0.02), C["upper_body2"] + _v(0, 0, -0.01), 0.105)
    cap("上半身2", "上半身2", C["upper_body2"] + _v(0, 0, 0.02), C["neck"] + _v(0, 0, -0.04), 0.115)
    cap("下半身", "下半身", _v(0, 0, 0.80), _v(0, 0, 0.76), 0.12)
    for sd, jp, sx in sides:
        def P(name, sd=sd):
            return C[f"{name}.{sd}"]
        rb.append(RigidBody(f"{jp}肩", f"{jp}肩", shape="sphere", size=(0.045, 0, 0), location=tuple(P("shoulder") + _v(0.02 * sx, 0, 0)),
                            mode="static"))
        cap(f"{jp}腕", f"{jp}腕", P("arm"), P("elbow"), 0.04)
        cap(f"{jp}ひじ", f"{jp}ひじ", P("elbow"), P("wrist"), 0.03)
        rb.append(RigidBody(f"{jp}手首", f"{jp}手首", shape="sphere", size=(0.045, 0, 0),
                            location=tuple(P("wrist") + (P("wrist") - P("elbow")) / 0.2 * 0.05), mode="static"))
        cap(f"{jp}足", f"{jp}足", P("leg"), P("knee"), 0.062)
        cap(f"{jp}ひざ", f"{jp}ひざ", P("knee"), P("ankle"), 0.048)
        cap(f"{jp}足首", f"{jp}足首", P("ankle") + _v(0, -0.02, -0.04), P("toe") + _v(0, 0, -0.005), 0.035)
    frames = SK.standard_frames(bones)
    frames["顔"] = []
    part = Part("mannequin", meshes=meshes, materials=mats, bones=bones, morphs=morph_decl, bodies=rb,
                frames={k: v for k, v in frames.items() if v},
                info={"landmarks": {**{k: v for k, v in C.items()}, "head_center": HEAD_C.copy()},
                      "head": {"center": HEAD_C.tolist(), "radii": HEAD_R.tolist()}})
    return part


def _chain(ctx, prefix, heads, parent, radius, group, root_mode="dynamic_bone"):
    """Bones, dynamic bodies and joints for one chain; heads[0..n-1] bone heads, heads[n] the tip."""
    bones, bodies, joints = [], [], []
    n = len(heads) - 1
    for i in range(n):
        bones.append(Bone(f"{prefix}{i + 1}", tuple(heads[i]), parent=parent if i == 0 else f"{prefix}{i}",
                          tail=tuple(heads[i + 1]) if i == n - 1 else None,
                          tail_bone="" if i == n - 1 else f"{prefix}{i + 2}", after_physics=True))
        bodies.append(RigidBody(f"{prefix}{i + 1}", f"{prefix}{i + 1}", shape="capsule",
                                **geo.capsule_between(heads[i], heads[i + 1], radius), group=group,
                                no_collide=(group,), mode=root_mode if i == 0 else "dynamic", mass=0.05,
                                damping=(0.9, 0.9)))
    anchor = ctx.find_body(parent)
    if anchor is not None:
        joints.append(Joint(f"{prefix}0", anchor.name, bodies[0].name, location=tuple(heads[0]),
                            rot_lo=(-0.6, -0.3, -0.6), rot_hi=(0.6, 0.3, 0.6), spring_rot=(20, 20, 20)))
    for i in range(1, n):
        joints.append(Joint(f"{prefix}{i}", bodies[i - 1].name, bodies[i].name, location=tuple(heads[i]),
                            rot_lo=(-0.7, -0.2, -0.7), rot_hi=(0.7, 0.2, 0.7), spring_rot=(10, 10, 10)))
    return bones, bodies, joints


@builder("mannequin_hair", needs=("mannequin",))
def build_hair(ctx):
    root = ctx.need("mannequin")
    cfg = ctx.cfg
    n_bones = int(cfg.get("bangs_bones", 3))
    z0, z1 = 1.448, 1.392                  # above the eyes (z 1.36): a short fringe
    zs = np.linspace(z0, z1, n_bones + 1)
    ys = _surface_point(0.0, zs)[..., 1] - np.linspace(0.0, 0.03, n_bones + 1)
    heads = np.stack([np.zeros(n_bones + 1), ys, zs], axis=1)
    bones, bodies, joints = _chain(ctx, "前髪", heads, "頭", 0.012, 4)

    def fn(u, v):
        x = (u - 0.5) * 0.13
        z = z0 + (z1 - z0) * v
        p = _surface_point(x, z + 0.0)
        p[..., 1] -= 0.004 + 0.03 * v
        return p
    sheet = geo.surface(fn, 8, 10)
    names = [b.name for b in bones]
    w = skin.chain_weights(sheet.verts, names, heads, blend=0.02)
    mesh = sheet.to_mesh("bangs", mats=["hair"], weights=skin.normalise(w, len(sheet.verts), fallback="頭"))
    mat = Material("hair", texture=_hair_texture(ctx), sphere=_hair_sphere(ctx), sphere_mode="add",
                   toon=_toon(ctx, "toon_hair", "#a04040"), double_sided=True, diffuse=(1, 1, 1, 1), edge=False)
    return Part("mannequin_hair", meshes=[mesh], materials=[mat], bones=bones, bodies=bodies, joints=joints,
                frames={"髪": names})


@builder("mannequin_skirt", needs=("mannequin",))
def build_skirt(ctx):
    root = ctx.need("mannequin")
    C = ctx.land
    top_z, bot_z = 0.88, 0.55
    fronts = np.array([[0, -0.115, 0.86], [0, -0.14, 0.76], [0, -0.215, 0.55]])
    backs = fronts * [1, -1, 1]
    bones, bodies, joints = [], [], []
    for pre, heads in (("スカート前", fronts), ("スカート後", backs)):
        b, rbs, js = _chain(ctx, pre, heads, "下半身", 0.035, 5, root_mode="dynamic")
        bones += b
        bodies += rbs
        joints += js
    profile = np.array([[0.215, bot_z], [0.17, 0.68], [0.13, 0.80], [0.115, top_z]])
    sk = geo.revolve(profile, segments=24)
    v = sk.verts
    wf = skin.chain_weights(v, ["下半身", "スカート前1", "スカート前2"],
                            [_v(0, -0.115, 0.9), fronts[0], fronts[1], fronts[2]], blend=0.07)
    wb = skin.chain_weights(v, ["下半身", "スカート後1", "スカート後2"],
                            [_v(0, 0.115, 0.9), backs[0], backs[1], backs[2]], blend=0.07)
    t = skin.plane_ramp(v, _v(0, 0, 0), _v(0, 1, 0), 0.06)            # 0 in front (-Y), 1 behind
    w = skin.normalise(skin.mix(wf, wb, t), len(v), fallback="下半身")
    mesh = sk.to_mesh("skirt", mats=["skirt"], weights=w)
    mat = Material("skirt", diffuse=(0.9, 0.35, 0.45, 1.0), ambient=(0.5, 0.25, 0.3), double_sided=True,
                   toon=_toon(ctx, "toon_skirt", "#c07a8e"))
    return Part("mannequin_skirt", meshes=[mesh], materials=[mat], bones=bones, bodies=bodies, joints=joints,
                frames={"スカート": [b.name for b in bones]})
