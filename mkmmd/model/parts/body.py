"""The body part: skin surface of an anime girl (torso, neck seam, arms, hands with five fingers, legs, feet), the MMD
standard skeleton from its landmarks, analytic skin weights, static colliders and the skin material with its generated
texture and toon ramp. Everything is driven by the spec: `[proportions]` (measured landmarks, torso cuts,
limb diameters, foot) and the optional `[body]` table (see below); other characters only need their own proportions.
The hands have their own design (body_hand.DESIGN), or come from a hand mesh asset (body_hand_mesh): their finger
joints are made from it, not read from the landmarks.

`[body]` keys (all optional):
  skin           {base, shade, blush, nail}  hex colours (default: [colors.skin] of the spec)
  nails          true | false | "#rrggbb"    nail plates on the fingertips (default true, pale pink); a colour paints them
  source         "procedural" (default: the body is built from the spec's proportions) | "mesh" (an artist's whole body
                 fitted to the skeleton, see body_donor.py: `mesh = "base:girl/body.npz"`) | "pmx" (taken from an existing
                 PMX model, see body_pmx.py: [body.pmx] path, materials, ...)
  resolution     {torso, arm, leg}           ring point counts (default 32, the hand's seam, 24; the arm must have as
                 many points as the hand's seam (32 for the designed hand): the hand is welded onto its last ring)
  hand           the hand's design, key by key over body_hand.DESIGN (length, palm, knuckles, fingers, width, taper,
                 depth, palm_depth, split, splay, curl, thumb, nail ...); or `mesh = "path/hand.npz"`, a left hand drawn
                 elsewhere (format: body_hand_mesh; a relative path is taken against the spec's folder, `base:girl/hand.npz`
                 names a model base's) with only length, bend and nail
  land, dims                                 overrides forwarded to body_shape.resolve
  skeleton       options of skeleton.standard_bones (e.g. {twist = "split"})

Published in `Part.info`: landmarks (every standard bone head, the extra points `toe_end`, `tail_root`, ... and the
mirrored right side), neck_top (the seam ring the head builds on), skin (colours, material and toon names),
torso_profile, regions (vertex ranges per shell), hand (frame and design), foot, colliders."""
import numpy as np

from .. import skeleton
from .. import spec as SP
from ..build import builder
from ..part import Material, Mesh, Part
from . import body_geom as G
from . import body_mesh as BM
from . import body_phys, body_shape, body_tex, body_weights

SKIN = "skin"
NAIL = "nail"
# material recipe shared with the face: diffuse 1.0 (the colour is in the texture), ambient 0.5, no specular, a toon ramp
# with a white lit half and a light pink shadow half
SKIN_AMBIENT = (0.5, 0.5, 0.5)
NAIL_AMBIENT = 0.42               # the nails' ambient as a share of their colour (see the nail material)


def _ring_point(rows, z_target, theta):
    """A point on the torso surface at height z and ring angle theta (0 = front, + towards +X): the rows are cuts."""
    z = np.array([r[0] for r in rows])
    hw = np.interp(z_target, z, [r[1] for r in rows])
    yf = np.interp(z_target, z, [r[2] for r in rows])
    yb = np.interp(z_target, z, [r[3] for r in rows])
    n = np.interp(z_target, z, [r[4] for r in rows])
    c, s = G.superellipse(np.array([theta]), n)
    return np.array([hw * s[0], 0.5 * (yf + yb) - 0.5 * (yb - yf) * c[0], z_target])


def extra_landmarks(shape, rows):
    """Non-bone points other parts want: the tail root and its normal on the lower back, the front of the chest."""
    L = shape.land
    out = {}
    zt = float(L["lower_body"][2] - 0.040)
    p = _ring_point(rows, zt, np.pi)
    out["tail_root"] = p
    eps = 0.004
    q1, q2 = _ring_point(rows, zt, np.pi + 0.05), _ring_point(rows, zt, np.pi - 0.05)
    qz = _ring_point(rows, zt + eps, np.pi) - _ring_point(rows, zt - eps, np.pi)
    nrm = np.cross(q1 - q2, qz)
    if nrm[1] < 0:
        nrm = -nrm
    for side, sx in (("L", 1.0), ("R", -1.0)):
        out[f"tail_root.{side}"] = _ring_point(rows, zt, np.pi - sx * 0.12) if False else p + np.array([sx * 0.022, 0.0, 0.0])
    out["bust_front"] = _ring_point(rows, float(L["upper_body2"][2] + 0.058), 0.0)
    return out, G.unit(nrm)


def smooth_normals(shape, shells, ranges, verts, faces, radius=0.026):
    """Smooth vertex normals across shells that overlap (pelvis/thighs, shoulder/arm): every vertex blends its normal
    with those of nearby vertices of OTHER shells that face the same way, so no shading crease shows where one shell
    meets another (the hand is welded to the arm, so its seam needs none). The top two rings of the torso get radial
    normals (the head's seam ring uses the same)."""
    n = G.vertex_normals(verts, faces)
    owner = np.zeros(len(verts), int)
    for k, sh in enumerate(shells):
        a, b = ranges[sh.name]
        owner[a:b] = k
    out = n.copy()
    zone = np.where(((verts[:, 2] < 0.84) & (verts[:, 2] > 0.60)) | ((verts[:, 2] > 1.02) & (verts[:, 2] < 1.20)))[0]
    for i0 in range(0, len(zone), 400):
        idx = zone[i0:i0 + 400]
        d = np.linalg.norm(verts[idx][:, None, :] - verts[None, :, :], axis=2)
        other = owner[idx][:, None] != owner[None, :]
        same_way = (n[idx] @ n.T) > 0.2
        w = np.where(other & same_way & (d < radius), 0.6 * (1.0 - d / radius) ** 2, 0.0)
        out[idx] = n[idx] + w @ n
    out = G.unit(out)
    nt = BM.neck_top(shape)
    top = np.abs(verts[:, 2] - nt["z"]) < 0.0055
    cx, cy = nt["center"]
    rad = np.stack([(verts[:, 0] - cx) / nt["rx"] ** 2, (verts[:, 1] - cy) / nt["ry"] ** 2, np.zeros(len(verts))], 1)
    out[top] = G.unit(rad[top])
    return out


def skin_look(ctx, cfg):
    """The skin's colours and material numbers: [colors.skin] under [body] skin and neck_shadow (both routes that draw
    their own skin use them)."""
    skin_cfg = SP.merge((ctx.spec.get("colors") or {}).get("skin") or {}, cfg.get("skin"))
    nk = cfg.get("neck_shadow") or {}
    if nk.get("color"):
        skin_cfg["neck_shadow"] = nk["color"]
    sk = cfg.get("skin") or {}
    return dict(cfg=skin_cfg, pal=body_tex.palette(skin_cfg),
                mult=np.array(sk.get("toon_shadow_multiplier", body_tex.DEFAULT_TOON), float),
                ambient=tuple(float(x) for x in sk.get("ambient", SKIN_AMBIENT)),
                specular=tuple(float(x) for x in sk.get("specular", (0.0, 0.0, 0.0))),
                shininess=float(sk.get("shininess", 0.0)), flush=float(sk.get("flush", body_tex.FLUSH)),
                neck=dict(height=float(nk.get("height", 0.045)), strength=float(nk.get("strength", 0.9))))


def skin_materials(ctx, cfg, look, tex_name, toon_name):
    """([skin material, + the nail material unless [body] nails = false], their names)."""
    nails = cfg.get("nails", True)
    edge = body_tex.hex_rgb(look["cfg"].get("outline", "#8a6a62"))
    mats = [Material(SKIN, name_en="skin", diffuse=(1.0, 1.0, 1.0, 1.0), specular=look["specular"],
                     shininess=look["shininess"], ambient=look["ambient"], texture=tex_name, toon=toon_name, edge=True,
                     edge_color=(float(edge[0]), float(edge[1]), float(edge[2]), 1.0), edge_size=0.6,
                     comment="body skin: generated atlas + toon ramp")]
    mat_names = [SKIN]
    if nails:
        accent = (ctx.spec.get("colors") or {}).get("accent") or {}
        if nails is True:
            ncol = look["cfg"].get("nail", "#f0b3ab")
        elif str(nails).lower() == "red":
            ncol = accent.get("nail_red", "#c9262d")
        else:
            ncol = str(nails)
        nc = body_tex.hex_rgb(ncol)
        # lit colour = ambient + 0.6 x diffuse (MMD's default light): an ambient of 0.42 x the colour shows the colour
        # itself instead of washing it out to white
        amb = tuple(float(x) for x in NAIL_AMBIENT * nc)
        mats.append(Material(NAIL, name_en="nails", diffuse=(float(nc[0]), float(nc[1]), float(nc[2]), 1.0),
                             specular=(0.20, 0.17, 0.17), shininess=30.0, ambient=amb, toon=toon_name,
                             edge=False, comment="fingernail plates, a separate material so the colour can change"))
        mat_names.append(NAIL)
    return mats, mat_names


def skin_info(look, tex_name, toon_name, z_top):
    """The published `skin` and `neck_shadow` (the shadow band ends at the seam, `z_top`)."""
    pal, mult, neck = look["pal"], look["mult"], look["neck"]
    return dict(
        skin=dict(base=body_tex.rgb_hex(pal["base"]), shade=body_tex.rgb_hex(pal["base"] * mult),
                  blush=body_tex.rgb_hex(pal["blush"]), blush_knee=body_tex.rgb_hex(pal["blush_knee"]),
                  toon_mult=tuple(float(x) for x in mult), ambient=look["ambient"], specular=look["specular"],
                  shininess=look["shininess"], tone=body_tex.rgb_hex(body_tex.base_tone(pal, look["flush"])),
                  toon=toon_name, texture=tex_name, material=SKIN),
        neck_shadow=dict(z_top=float(z_top), height=neck["height"], strength=neck["strength"],
                         color=body_tex.rgb_hex(pal["neck_shadow"]), front=1.0, back=0.35,
                         note="mix(skin, color, strength * (back + (front - back) * (0.5 + 0.5 cos(theta)) ** 1.5)) at the seam, "
                              "smoothstep to the skin over `height` metres below it; theta 0 = front, + towards her left"))


def neck_ring(nt, n=32):
    """(n, 3) points of the seam ellipse `nt` (neck_top): point k at 2 pi k / n from the front (-Y) towards +X."""
    th = 2 * np.pi * np.arange(n) / n
    c, s = G.superellipse(th, 2.0)
    return np.stack([nt["center"][0] + nt["rx"] * s, nt["center"][1] - nt["ry"] * c, np.full(n, nt["z"])], 1)


@builder("body")
def build(ctx):
    cfg = ctx.cfg or {}
    source = cfg.get("source", "procedural")
    if source == "pmx":
        from .body_pmx import build_pmx
        return build_pmx(ctx)
    if source == "mesh":
        from .body_donor import build_donor
        return build_donor(ctx)
    if source != "procedural":
        raise ValueError(f"[body] source must be 'procedural', 'mesh' or 'pmx', got {source!r}")
    prop = ctx.spec.get("proportions") or {}
    shape = body_shape.resolve(prop, cfg, base=getattr(ctx.spec, "dir", None))
    for n in shape.notes:
        ctx.log("WARNING " + n)
    res = cfg.get("resolution") or {}
    nails = cfg.get("nails", True)
    shells = BM.build_shells(shape, int(res.get("torso", 32)), res.get("arm"), int(res.get("leg", 24)),
                             nails=bool(nails))
    verts, faces, uvs, face_mat, ranges, s_par, th_par, ring_par, gidx = G.join(shells)
    nv = len(verts)

    # ---- skeleton
    land = {k: np.asarray(v, float) for k, v in shape.land.items()}
    bones = skeleton.standard_bones(land, cfg.get("skeleton"))
    deform = {b.name for b in bones if b.deform}
    names = {b.name for b in bones}

    # ---- weights
    weights = body_weights.body_weights(shape, shells, ranges, gidx, nv, deform)

    # ---- materials and textures
    look = skin_look(ctx, cfg)
    torso = shells[0]
    z0, z1 = torso.s_range
    neck = dict(look["neck"], v_h=look["neck"]["height"] / (z1 - z0))
    tex = body_tex.skin_texture(body_tex.marks(shells), look["pal"], flush=look["flush"], neck=neck)
    toon = body_tex.toon_ramp(look["pal"], look["mult"])
    tex_name = ctx.save_png("skin", tex)
    toon_name = ctx.save_png("skin_toon", toon)
    mats, mat_names = skin_materials(ctx, cfg, look, tex_name, toon_name)

    normals = smooth_normals(shape, shells, ranges, verts, faces)
    mesh = Mesh("body", verts=verts, faces=faces, uv=np.concatenate(uvs, 0), face_mat=face_mat, mats=mat_names,
                weights=weights, normals=normals, smooth=True)

    # ---- colliders
    bodies = body_phys.make_bodies(shape, shells, names)

    # ---- published information
    rows = BM.torso_rows(shape)
    extra, tail_n = extra_landmarks(shape, rows)
    landmarks = dict(shape.land)
    landmarks.update(extra)
    nt = BM.neck_top(shape)
    N = 32
    info = dict(
        landmarks=landmarks,
        tail_normal=tail_n,
        neck_top=dict(nt, ring=neck_ring(nt, N), n=N, start="front", dir="ccw_from_above"),
        **skin_info(look, tex_name, toon_name, z1),
        torso_profile={"rows": np.array([r[0] for r in rows]), "half_width": np.array([r[1] for r in rows]),
                       "y_front": np.array([r[2] for r in rows]), "y_back": np.array([r[3] for r in rows]),
                       "cuts": np.array(rows)},
        regions=dict(ranges),
        hand=dict(frame=tuple(np.asarray(x) for x in shape.frame), design=shape.hand),
        foot=dict(shape.dims["foot"]),
        mesh="body",
    )
    return Part("body", meshes=[mesh], materials=mats, bones=bones, bodies=bodies,
                frames=skeleton.standard_frames(bones), info=info)
