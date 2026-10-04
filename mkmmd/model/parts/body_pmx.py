"""The body part in `pmx` mode: the character's body and skeleton are taken from an existing PMX model (`pmx_take.take`).

Spec (`[body]` in body.toml): `source = "pmx"` and the table `[body.pmx]`:

  path        the PMX file (`~` and paths relative to the spec's folder work)
  materials   every material to take, face and body together (the head part builds its face from the same Take)
  skin        the materials that make up the body surface (default: the first of `materials`); their triangles are re-sorted
              into the visible `skin` material and the covered one `skin_hidden` (see `hide`)
  scale       uniform scale about the floor (default: `[proportions] s_body`)
  head_scale  the head's absolute scale (default: `[proportions] s_head`); the head, its eyes and everything weighted to the
              head bone's subtree grow from `scale` to `head_scale` by weight, so the neck blends
  head_bone   the head bone's name (default 頭)
  unit        metres per PMX unit (default 0.08)
  names       {skin = "skin", hidden = "skin_hidden"}: the names of the two body materials
  skip_morphs, keep_bones, drop_bones, recolor    forwarded to `pmx_take` (recolor: rules for the texture copies)
  landmark_bones   {toe = "足つま先"}: extra standard-like bones the standard map lacks (stem -> bone name without side)
  head_sphere      [cx, cy, cz, r]: the head collider instead of the fitted one. Set it whenever the hair hangs close to the
                   face: the fit only sees the face shell, which is open at the back, and overshoots the forehead
  [body.pmx.hide]  the skin an outfit covers in every pose (moved to the hidden material, alpha 0, still weighted and part of
                   the mesh so garments fit and transfer weights from the whole body): groups (default torso, shoulder, arm),
                   neck_z (or collar_z + neck_inside = 0.03: the neck skin stays visible down to that height, inside the
                   collar), wrist_back (m, default 0.035: the sleeves end this far before the wrist), leg_z (legs above it
                   are covered; below it everything shows)

The surface keeps the source's materials exactly (colours, toon, sphere, edge, flags); the skeleton is the source's, pruned.
Published in `Part.info`: take (the `pmx_take.Take`, also as `base`), landmarks (+ tail_root, tail_normal, bust_front),
neck_top, torso_profile, skin (material names, texture, toon), regions, hidden (counts), colliders are `col_*` bodies."""
import dataclasses

import numpy as np

from ...core import bonemap
from .. import pmx_take
from ..build import BuildError
from ..part import Material, Mesh, Morph, Part
from . import body_pmx_fit as F


def _hex_to_f(c):
    return tuple(float(x) for x in c)


def resolve_hide(hide, prop):
    """The `hide` table with `neck_z` derived from the outfit's collar height when only that is known."""
    h = dict(hide or {})
    if h.get("neck_z") is None:
        collar = h.get("collar_z")
        if collar is None:
            collar = ((prop.get("outfit_guides") or {}).get("collar_top_z"))
        if collar is not None:
            h["neck_z"] = float(collar) - float(h.get("neck_inside", 0.03))
    return h


def check_required(bones):
    smap = bonemap.build_map({b.name: b.name for b in bones})
    missing = [s for s in bonemap.REQUIRED if s not in smap]
    if missing:
        raise BuildError(f"the imported skeleton lacks the standard bones {missing} (kept: {len(bones)} bones)")
    return smap


def compare_with_spec(land, prop, log, tol=0.003, skip=("eye", "toe")):
    """Warn when a landmark of the spec's [proportions.landmarks] and the imported bone disagree by more than `tol` metres
    (the eye and toe landmarks are skipped: the head part owns the eye pivots, and the foot tip is a convention)."""
    want = prop.get("landmarks") or {}
    bad = []
    for k, v in want.items():
        if any(s in k for s in skip):
            continue
        if k in land and np.linalg.norm(np.asarray(v, float) - land[k]) > tol:
            bad.append((k, round(float(np.linalg.norm(np.asarray(v, float) - land[k])) * 1000, 1)))
    if bad:
        log("WARNING imported bones differ from [proportions.landmarks] (mm): " + ", ".join(f"{k} {d}" for k, d in bad[:8]))
    return bad


def build_pmx(ctx):
    cfg = ctx.cfg or {}
    pc = dict(cfg.get("pmx") or {})
    prop = ctx.spec.get("proportions") or {}
    if not pc.get("path") or not pc.get("materials"):
        raise BuildError("body source = 'pmx' needs [body.pmx] path and materials")
    kept = list(pc["materials"])
    skin_src = list(pc.get("skin") or kept[:1])
    for m in skin_src:
        if m not in kept:
            raise BuildError(f"[body.pmx] skin material {m!r} is not in materials")
    s_body = float(pc.get("scale", prop.get("s_body", 1.0)))
    s_head = float(pc.get("head_scale", prop.get("s_head", s_body)))
    head_bone = pc.get("head_bone", "頭")
    tk = pmx_take.take(pc["path"], kept, unit=float(pc.get("unit", 0.08)), scale=s_body,
                       head={"bone": head_bone, "factor": s_head / s_body}, skip_morphs=pc.get("skip_morphs", ()),
                       keep_bones=pc.get("keep_bones", ()), drop_bones=pc.get("drop_bones", ()),
                       base_dir=getattr(ctx.spec, "dir", None))
    bones = tk.bones
    smap = check_required(bones)
    inv = {b: s for s, b in smap.items()}
    for b in bones:
        b.semantic = inv.get(b.name, "")
    names = {b.name for b in bones}
    land = F.landmarks(bones, pc.get("landmark_bones"))
    compare_with_spec(land, prop, ctx.log)

    # ---- the body surface and its regions
    piece = tk.piece(skin_src)
    V, nv = piece.verts, len(piece.verts)
    tris = np.array(piece.faces, int).reshape(-1, 3)
    groups = F.bone_groups([b.name for b in bones])
    G = F.vertex_groups(piece.weights, groups, nv)
    fg = F.face_groups(tris, G)
    hide = resolve_hide(pc.get("hide"), prop)
    hidden = F.hidden_faces(V, tris, G, land, hide)
    ctx.log(f"body: {nv} vertices, {len(tris)} triangles, {int(hidden.sum())} covered by the outfit "
            f"({100.0 * hidden.mean():.0f} %), {len(bones)} bones")

    # ---- materials: the source's own skin material, visible, and a copy that is alpha 0 for the covered skin
    nm = {"skin": "skin", "hidden": "skin_hidden", **(pc.get("names") or {})}
    base = pmx_take.part_materials(tk, ctx, [skin_src[0]], recolor=pc.get("recolor"))[skin_src[0]]
    vis = dataclasses.replace(base, name=nm["skin"])
    hid = dataclasses.replace(base, name=nm["hidden"], diffuse=(*base.diffuse[:3], 0.0), edge=False, alpha_blend=True,
                              comment="skin covered by the outfit in every pose: invisible, still part of the body")
    mesh = Mesh("body", verts=V, faces=piece.faces, uv=piece.uv, face_mat=hidden.astype(int), mats=[vis.name, hid.name],
                weights=piece.weights, morphs=piece.morphs, normals=piece.normals, smooth=True)
    declared = [Morph(m.name, m.panel, m.name_en) for m in tk.morphs if m.kind == "vertex" and m.name in piece.morphs]

    # ---- colliders and the points the other parts ask for
    head_v = tk.verts[tk.head_weight > 0.5]
    hs = pc.get("head_sphere")
    bodies = F.fit_bodies(V, G, land, head_v, names, head_hint=None if hs is None else (hs[:3], hs[3]))
    extra, tail_n = F.extra_landmarks(V, tris, piece.normals, land, fg)
    landmarks = dict(land)
    landmarks.update(extra)
    z_neck = float(land["head"][2]) - 0.040 * s_body                       # the plain neck ring: just under the chin
    nz = V[G[:, F.GROUPS.index("neck")] > 0.5, 2]
    if len(nz):
        z_neck = min(z_neck, float(nz.max()) - 0.003)                      # ... but inside the neck skin the body owns
    info = dict(
        landmarks=landmarks, tail_normal=tail_n, take=tk, base=tk, mesh="body", source="pmx",
        neck_top=F.neck_ring(V, tris, fg, z_neck),
        torso_profile=F.torso_profile(V, tris, fg, float(land["lower_body"][2]) - 0.02, float(land["neck"][2])),
        skin=dict(material=vis.name, hidden=hid.name, texture=vis.texture, toon=vis.toon, sphere=vis.sphere,
                  ambient=vis.ambient, specular=vis.specular, shininess=vis.shininess),
        regions={g: np.where(G.sum(1) > 0.5)[0][np.argmax(G[G.sum(1) > 0.5], axis=1) == k] for k, g in enumerate(F.GROUPS)},
        hidden=dict(faces=int(hidden.sum()), total=len(tris), rules=hide),
        height=float(tk.verts[:, 2].max()), bone_groups=groups)
    return Part("body", meshes=[mesh], materials=[vis, hid], bones=bones, bodies=bodies, morphs=declared,
                frames={k: [n for n in v if n in names] for k, v in tk.frames.items()}, info=info)
