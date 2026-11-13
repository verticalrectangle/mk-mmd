"""The head part in body mode (`[head] source = "body"`): the face is part of the body import (`pmx_take`, done by the body part and
published as `info["take"]`); this part builds the face meshes from it, closes the back of the skull, and publishes what the
hair and the outfit fit to.

What it does with the take (`[head.take]` in the spec):
  materials   the face materials (skin, sclera, highlights, iris, mouth, tongue, teeth, lashes + brows, morph-only pieces); one
              mesh each, the skin one named `face` (the hair part binds its forehead shadow to it)
  skin        the material that is the skin (default the first); its boundary: the outer loop (where the source model's own hair
              began), the neck (joined natively to the body skin by shared ring vertices: nothing to stitch) and the openings
  recolor     [[head.take.recolor]] rules for copies of the textures (pmx_recolor): iris, lashes, brows ...
  eyes        {left = "<left eye bone>", right = "<right eye bone>", iris = "<iris material>"}   (published eye data)
  brow_bone / brow_material   the bone that carries the brows (their vertices are found by its weights) and their material
  head_bone   the head bone (default 頭): the generated cap and ears hang on it
  cap         {rings, fade, pole_gap}: the skull cap (head_cap.build_cap)
  shell_deg   resolution of the radial map used for the published surface (default 2)

`[head.edit]` (head_edit): small smooth edits of the imported face that keep its morphs working: eyes (the outer corners tilted
up), jaw (a softer cheek and jaw line), fang (one canine behind the upper lip), highlight (the main highlight narrowed), pupil
(texture rules applied after [head.take.recolor]: a vertical almond pupil, a narrower highlight in the iris texture).
Each sub-table is switched off with `enabled = false` (or all with `[head.edit] enabled = false`).

The imported face is a front shell: it ends in an open boundary at the back and the top, where the source's hair began. A
cap, generated from the parametric skull (head_shape) and starting exactly on that boundary, closes it (hidden under the hair,
and what the hair is fitted to); a radial map of the closed shell (head_shell.RadialShape) stands in for the implicit head
shape when the hairline, the ear anchors and the face outline are computed. Ears are generated on the shell.
"""
from pathlib import Path

import numpy as np

from .. import pmx_take as TK
from .. import spec as SP
from ..part import Mesh, Morph, Part
from . import head_cap as CP
from . import head_ear as HR
from . import head_edit as HE
from . import head_shape as HS
from . import head_shell as SH
from . import head_tex as TX

HEAD = "頭"


# ------------------------------------------------------------------------------------------------------ texture patches
def uv_occupancy(polys, res=256):
    """Boolean (res, res) image of the texels (cell centres) covered by the polygons `polys` (list of (n, 2) uv, v from the top)."""
    occ = np.zeros((res, res), bool)
    for pts in polys:
        pts = np.asarray(pts, float)
        x, y = pts[:, 0] * res, pts[:, 1] * res
        x0, x1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()), res))
        y0, y1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()), res))
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        n = len(pts)
        cr = np.stack([(x[(k + 1) % n] - x[k]) * (gy - y[k]) - (y[(k + 1) % n] - y[k]) * (gx - x[k]) for k in range(n)])
        occ[y0:y1, x0:x1] |= (cr >= -1e-9).all(0) | (cr <= 1e-9).all(0)
    return occ


def free_blocks(occ, sizes, margin=1):
    """Top-left cells of free rectangles `sizes` = [(cells_w, cells_h), ...] (each with a free margin), scanning from the
    bottom right of the occupancy image."""
    res = occ.shape[0]
    taken = occ.copy()
    out = []
    for cw, ch in sizes:
        found = None
        for y in range(res - ch - 2 * margin, -1, -1):
            for x in range(res - cw - 2 * margin, -1, -1):
                if not taken[y:y + ch + 2 * margin, x:x + cw + 2 * margin].any():
                    found = (x + margin, y + margin)
                    break
            if found:
                break
        if found is None:
            raise ValueError("the skin texture has no free space for the generated patches")
        taken[found[1] - margin:found[1] + ch + margin, found[0] - margin:found[0] + cw + margin] = True
        out.append(found)
    return out


def _samples(img, uv):
    """RGB (0..255) of the texels under each uv (n, 2) (v from the top)."""
    h, w = img.shape[:2]
    x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    y = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
    return img[y, x, :3].astype(float)


def ring_index(n, faces, start):
    """Edge rings of a mesh's vertices counted from the vertices `start` (0 there, 1 next to them ...; -1 where none
    reaches)."""
    nbr = [set() for _ in range(n)]
    for f in faces:
        for k in range(len(f)):
            a, b = int(f[k]), int(f[(k + 1) % len(f)])
            nbr[a].add(b)
            nbr[b].add(a)
    ring = np.full(n, -1)
    front = [int(i) for i in start]
    ring[front] = 0
    r = 0
    while front:
        r += 1
        nxt = [b for a in front for b in nbr[a] if ring[b] < 0]
        nxt = list(dict.fromkeys(nxt))
        ring[nxt] = r
        front = nxt
    return ring


def surface_distance(verts, faces, start):
    """Shortest distance along the mesh's edges from the vertices `start` to every vertex (inf where none reaches), and
    the start vertex each one is nearest to (-1 where none reaches)."""
    import heapq
    n = len(verts)
    nbr = [dict() for _ in range(n)]
    for f in faces:
        for k in range(len(f)):
            a, b = int(f[k]), int(f[(k + 1) % len(f)])
            d = float(np.linalg.norm(verts[a] - verts[b]))
            nbr[a][b] = nbr[b][a] = d
    dist, owner = np.full(n, np.inf), np.full(n, -1)
    heap = [(0.0, int(i)) for i in start]
    dist[[int(i) for i in start]] = 0.0
    owner[[int(i) for i in start]] = [int(i) for i in start]
    while heap:
        d, a = heapq.heappop(heap)
        if d > dist[a]:
            continue
        for b, e in nbr[a].items():
            if d + e < dist[b]:
                dist[b], owner[b] = d + e, owner[a]
                heapq.heappush(heap, (d + e, b))
    return dist, owner


def _raster(shape, polys, values=None, window=None):
    """Rasterise convex polygons (uv, v from the top) at texel centres into an image of `shape` (h, w), inside `window`
    (x0, y0, x1, y1). Without `values`: a coverage mask (bool). With `values` (per polygon, (corners, c)): the values
    interpolated over a fan of triangles at each texel, (h, w, c); where polygons overlap the one whose first channel is
    smallest wins; inf where nothing lies."""
    h, w = shape
    x0, y0, x1, y1 = window or (0, 0, w, h)
    if values is None:
        out = np.zeros((y1 - y0, x1 - x0), bool)
    else:
        out = np.full((y1 - y0, x1 - x0, np.shape(values[0])[1]), np.inf)
    for k, pts in enumerate(polys):
        x, y = pts[:, 0] * w - x0, pts[:, 1] * h - y0
        a0, a1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()), x1 - x0))
        b0, b1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()), y1 - y0))
        if a1 <= a0 or b1 <= b0:
            continue
        qx, qy = np.meshgrid(np.arange(a0, a1) + 0.5, np.arange(b0, b1) + 0.5)
        for j in range(1, len(pts) - 1):                                      # a fan of triangles
            ax, ay, bx, by, cx, cy = x[0], y[0], x[j], y[j], x[j + 1], y[j + 1]
            det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(det) < 1e-12:
                continue
            l1 = ((by - cy) * (qx - cx) + (cx - bx) * (qy - cy)) / det
            l2 = ((cy - ay) * (qx - cx) + (ax - cx) * (qy - cy)) / det
            l3 = 1.0 - l1 - l2
            inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
            if values is None:
                out[b0:b1, a0:a1] |= inside
                continue
            v = np.asarray(values[k], float)
            val = l1[..., None] * v[0] + l2[..., None] * v[j] + l3[..., None] * v[j + 1]
            cur = out[b0:b1, a0:a1]
            win = inside & (val[..., 0] < cur[..., 0])
            cur[win] = val[win]
    return out


def scalp_tint(img, uv_px, verts, faces, loop, other_polys, color, band=6, margin=3):
    """The hairline band of an imported face shell turned into `color`, in a copy of the skin texture (RGBA uint8). Such
    a shell is often painted in a tint of its source's hair along its outer boundary (where that hair began), fading
    into the skin a few edge rings in; under other hair that tint shows through. Every texel of the skin's faces takes
    `color` by a weight that is 1 on the boundary and fades smoothly to 0 at the distance (along the surface) of the
    vertices `band` edge rings in, mixed into the texel with the source's tint taken out: its share of the line from the
    skin there (the colour of the nearest vertex of that inner ring) to the tint (the median colour of the boundary). So
    the boundary is `color`, the band fades into the skin as it is painted beyond the band, with no trace of the old
    tint, and a piece of the skin the boundary does not reach keeps its colours. The `margin` texels round the faces
    (not on another piece: `other_polys`) follow their neighbours, so filtering shows no old colour at the edge. `uv_px`
    (n, 2) the skin's vertex uv with v from the top, `verts` (n, 3), `faces` its faces, `loop` the boundary's vertex
    ids. Returns (image, skin colour (3,) 0..255: the inner ring's median)."""
    out = np.array(img, copy=True)
    h, w = out.shape[:2]
    dist, _ = surface_distance(verts, faces, loop)
    rings = ring_index(len(verts), faces, loop)
    inner = np.nonzero(rings == min(band, rings.max()))[0]
    reach = max(float(np.median(dist[inner])), 1e-6)
    tint = np.median(_samples(out, uv_px[loop]), axis=0)
    ring_cols = _samples(out, uv_px[inner])
    skin = np.median(ring_cols, axis=0)
    _, owner = surface_distance(verts, faces, inner)
    row = np.full(len(verts), -1)
    row[inner] = np.arange(len(inner))
    skin_v = np.where((owner >= 0)[:, None], ring_cols[row[np.maximum(owner, 0)]], skin)
    # the old tint is taken out a little beyond the band
    far = 1.6 * reach
    near = [f for f in faces if min(dist[int(i)] for i in f) < far]
    if not near:
        return out, skin
    polys = [uv_px[list(f)] for f in near]
    allp = np.concatenate(polys, 0) * [w, h]
    win = (int(max(allp[:, 0].min() - margin - 1, 0)), int(max(allp[:, 1].min() - margin - 1, 0)),
           int(min(allp[:, 0].max() + margin + 2, w)), int(min(allp[:, 1].max() + margin + 2, h)))
    R = _raster((h, w), polys, [np.column_stack([dist[list(f)], skin_v[list(f)]]) for f in near], win)
    own = np.isfinite(R[..., 0])
    keep = own | (_raster((h, w), other_polys, window=win) & ~own)
    # the margin takes its nearest neighbour's values
    for _ in range(margin):
        for sl_to, sl_from in (((slice(1, None),), (slice(None, -1),)), ((slice(None, -1),), (slice(1, None),)),
                               ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                               ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
            src = R[sl_from]
            dst = R[sl_to]
            take = (src[..., 0] < dst[..., 0]) & ~keep[sl_to]
            dst[take] = src[take]
    gy, gx = np.nonzero(np.isfinite(R[..., 0]) & (R[..., 0] < far))
    d = R[gy, gx, 0]
    # the new colour: 1 on the boundary, 0 at the reach
    wt = 1.0 - HE.smooth3(d / reach)
    # the old tint's removal, fading out by `far`
    out_w = 1.0 - HE.smooth3((d - 0.5 * reach) / (far - 0.5 * reach))
    base = R[gy, gx, 1:]
    gy, gx = gy + win[1], gx + win[0]
    rgb = out[gy, gx, :3].astype(float)
    axis = tint[None] - base
    a = np.clip(((rgb - base) * axis).sum(1) / np.maximum((axis * axis).sum(1), 1e-6), 0.0, 1.0)
    plain = rgb - (out_w * a)[:, None] * axis                                # the old tint taken out
    new = plain * (1.0 - wt[:, None]) + TK.hex_rgb(color) * 255.0 * wt[:, None]
    out[gy, gx, :3] = np.clip(new + 0.5, 0, 255).astype(np.uint8)
    return out, skin


def _normals(V, faces):
    N = np.zeros_like(V)
    for f in faces:
        for k in range(1, len(f) - 1):
            n = np.cross(V[f[k]] - V[f[0]], V[f[k + 1]] - V[f[0]])
            for i in (f[0], f[k], f[k + 1]):
                N[i] += n
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def _orient_faces(V, faces, ref_dir):
    """Flip faces whose normal has a negative dot with ref_dir(mid) (the outward direction there)."""
    out = []
    for f in faces:
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        mid = V[list(f)].mean(0)
        out.append(tuple(f) if n @ ref_dir(mid) >= 0 else tuple(reversed(f)))
    return out


def azimuth_xy(P, c):
    """Azimuth about the vertical axis through c (0 straight ahead, positive towards +x) of points P (n, 3)."""
    return np.arctan2(P[:, 0] - c[0], -(P[:, 1] - c[1]))


def _grad_uv(P, t, patch, axis_xy):
    """Per-corner uv (k, 2; v up) of a face with vertices P (k, 3, head-local) and rows t (k,) in 0..1 (0 = bottom row) of a
    gradient patch (columns = azimuth about axis_xy). A vertex on the axis (the crown) takes the mean azimuth of the others; the
    face is shifted by whole turns so that it lies inside the patch's padded azimuth range."""
    az = azimuth_xy(P, axis_xy)
    on_axis = np.hypot(P[:, 0] - axis_xy[0], P[:, 1] - axis_xy[1]) < 1e-6
    ref = az[~on_axis][0] if (~on_axis).any() else 0.0
    th = ref + (az - ref + np.pi) % (2 * np.pi) - np.pi                        # contiguous round the back seam
    if on_axis.any() and (~on_axis).any():
        th[on_axis] = th[~on_axis].mean()
    th = th - 2 * np.pi * np.round(th.mean() / (2 * np.pi))
    th = np.clip(th, patch["th0"], patch["th1"])
    t = np.clip(t, 0.0, 1.0)
    u = (patch["x0"] + (th - patch["th0"]) / (patch["th1"] - patch["th0"]) * (patch["w"] - 1) + 0.5) / patch["W"]
    v = (patch["y0"] + (1.0 - t) * (patch["h"] - 1) + 0.5) / patch["H"]
    return np.stack([u, 1.0 - v], -1)


def _fan(f):
    return [(int(f[0]), int(f[k]), int(f[k + 1])) for k in range(1, len(f) - 1)]


# ------------------------------------------------------------------------------------------------------------- the build
def _loop_axis(V, loop):
    """Centre (x, y) and half extents of a boundary loop: the neck tube's ellipse."""
    P = V[loop[1]]
    lo, hi = P.min(0), P.max(0)
    return 0.5 * (lo[:2] + hi[:2]), 0.5 * (hi[:2] - lo[:2])


def _neck_info(body, V, neck, pivot):
    """The neck_top description (z, centre, rx, ry, ring (n, 3) in MODEL space): the body's when it publishes one, else the
    ellipse of the skin's own neck boundary."""
    nt = (body.info or {}).get("neck_top")
    if nt is not None and nt.get("ring") is not None:
        return nt
    c, r = _loop_axis(V, neck)
    z = float(V[neck[1]][:, 2].mean())
    n = 32
    a = 2 * np.pi * np.arange(n) / n
    ring = np.stack([c[0] + r[0] * np.sin(a), c[1] - r[1] * np.cos(a), np.full(n, z)], -1) + pivot
    return dict(z=z + pivot[2], center=(float(c[0] + pivot[0]), float(c[1] + pivot[1])), rx=float(r[0]), ry=float(r[1]), ring=ring)


def build(ctx):
    from . import head as H
    cfg = ctx.cfg or {}
    tc = SP.merge({}, cfg.get("take"))
    prop = ctx.spec.get("proportions") or {}
    body = ctx.need("body")
    tk = (body.info or {}).get("take")
    if tk is None:
        raise ValueError('[head] source = "body" needs the body part to publish info["take"] (the imported base, pmx_take.take)')
    names = list(tc.get("materials") or ())
    if not names:
        raise ValueError("[head.take] materials: list the face materials to build")
    skin_name = tc.get("skin") or names[0]
    head_bone = tc.get("head_bone", HEAD)
    pivot = np.asarray(tk.bone(head_bone).head, float)
    frame = H.Frame(pivot, 1.0)
    pieces = {n: tk.piece([n]) for n in names}
    skin = pieces[skin_name]
    sk_mat = tk.materials[skin_name]
    panel_of = {m.name: m.panel for m in tk.morphs if m.kind == "vertex"}
    mixed = _mix_morphs(pieces, cfg.get("morphs") or (), panel_of)

    # ---- boundary loops of the skin shell (head-local)
    V = skin.verts - pivot
    loops = CP.boundary_loops(V, skin.faces)
    if len(loops) < 2:
        raise ValueError(f"the skin material {skin_name!r} needs an outer boundary and a neck boundary")
    size = lambda l: float(np.ptp(V[l[1]], axis=0).sum())
    outer = max(loops, key=size)
    rest = [l for l in loops if l is not outer]
    neck = min([l for l in rest if len(l[0]) >= 8], key=lambda l: V[l[1]][:, 2].mean())
    holes = [l for l in rest if l is not neck]
    nt = _neck_info(body, V, neck, pivot)
    nc = frame.to_local(np.array([nt["center"][0], nt["center"][1], nt["z"]]))
    shape = HS.HeadShape(H.anchors_from(prop, frame, cfg.get("shape")),
                         neck=dict(c=(0.0, float(nc[1]), 0.0), rx=nt["rx"], ry=nt["ry"], k=0.014, top=0.02))
    centre = shape.centre
    ed = cfg.get("edit") or {}
    edit_on = lambda k: bool(ed.get("enabled", True)) and isinstance(ed.get(k), dict) and bool(ed[k].get("enabled", True))
    mouth = [h for h in holes if abs((V[h[1]] + pivot)[:, 0].mean()) < 0.01]
    ec = dict(left="左目", right="右目")
    ec.update(tc.get("eyes") or {})
    eye_geo = _eye_geometry(tk, ec, holes, V, pivot)
    if edit_on("eyes") and eye_geo:
        ecfg = dict(ed["eyes"])
        ecfg.setdefault("tilt_deg", (prop.get("face") or {}).get("eye_outer_tilt_deg", 4.0))
        HE.tilt_eyes(pieces, skin_name, {s: (g["centre"], g["width"]) for s, g in eye_geo.items()}, ecfg, tc.get("brow_bone"))
    if edit_on("jaw") and eye_geo and mouth:
        ez = min(float(g["opening"][:, 1].min()) for g in eye_geo.values())
        HE.soften_jaw(skin, ez, float(V[neck[1]][:, 2].mean() + pivot[2]), float(centre[1] + pivot[1]), ed["jaw"])
    if edit_on("highlight") and ed["highlight"].get("material") in pieces:
        HE.narrow_highlight(pieces[ed["highlight"]["material"]], float(ed["highlight"].get("scale_x", 0.8)))
    for key, default in (("lids", "まばたき"), ("mouth", "口角下げ")):
        if edit_on(key):
            _bake(ctx, pieces, ed[key], default, panel_of, key)
    if edit_on("move"):
        _move(pieces, ed["move"])
    V = skin.verts - pivot                                                       # the edited skin (the loops keep their vertex ids)
    turn = np.unwrap(CP.azimuth(V[outer[1]], centre))
    if abs(abs(turn[-1] - turn[0]) - 2 * np.pi) > 1.2:
        ctx.log(f"WARNING the skin's outer boundary winds {np.degrees(turn[-1] - turn[0]):.0f} deg round the head axis, not once")

    # ---- the skull cap: rings from the skin's outer boundary to the crown
    cap_cfg = dict(rings=9, fade=0.72, pole_gap=0.10)
    cap_cfg.update(tc.get("cap") or {})
    nl = len(outer[1])
    cap_new, cap_local = CP.build_cap(V[outer[1]], centre, shape.surface, **cap_cfg)
    nV = len(V)
    glob = np.concatenate([outer[1], nV + np.arange(len(cap_new))])
    all_loc = np.concatenate([V, cap_new], 0)
    cap_faces = _orient_faces(all_loc, [tuple(int(glob[i]) for i in f) for f in cap_local], lambda m: m - centre)
    n_all = len(all_loc)

    # ---- generated colour patches, painted into a free corner of the skin texture: a flat colour for the ears and a gradient for the
    # cap that starts in the colours the shell ends in (so the seam does not show) and relaxes to their mean at the
    # crown, or to [head.take.scalp] crown; with [head.take.scalp] color the shell's own hairline band is tinted first
    # (scalp_tint)
    uv_px = np.stack([skin.vert_uv[:, 0], 1.0 - skin.vert_uv[:, 1]], -1)       # v from the top
    cp = dict(width=96, rows=16, pad=0.35)
    cp.update(tc.get("patch") or {})
    sc = tc.get("scalp") or {}
    patch = {}

    def paint(img):
        polys, own = [], []
        for name, pm in pieces.items():
            tx = tk.materials[name].texture
            if tx is not None and tx.read_bytes() == sk_mat.texture.read_bytes():
                c = 0
                for fc in pm.faces:
                    q = pm.uv[c:c + len(fc)]
                    polys.append(np.stack([q[:, 0], 1.0 - q[:, 1]], -1))
                    own.append(name == skin_name)
                    c += len(fc)
        out = np.array(img, copy=True)
        skin_col = None
        if sc.get("color"):
            others = [p for p, o in zip(polys, own) if not o]
            out, skin_col = scalp_tint(out, uv_px, skin.verts, skin.faces, outer[1], others, sc["color"],
                                       int(sc.get("band", 6)))
        h, w = out.shape[:2]
        res = min(256, w, h)
        cell = w // res
        wpx, hpx = min(int(cp["width"]), w // 2), min(int(cp["rows"]), max(h // 4, 2))
        gsize = (-(-wpx // cell), -(-hpx // cell))
        (cx, cy), (gx, gy) = free_blocks(uv_occupancy(polys, res), [(4, 4), gsize])
        loop_cols = _samples(out, uv_px[outer[1]])
        col = loop_cols.mean(0)
        flat = col if skin_col is None else skin_col                             # a tinted hairline: the ears stay skin
        x0, y0 = cx * cell + cell // 2, cy * cell + cell // 2
        out[y0:y0 + 3 * cell, x0:x0 + 3 * cell, :3] = np.clip(flat + 0.5, 0, 255).astype(np.uint8)
        out[y0:y0 + 3 * cell, x0:x0 + 3 * cell, 3] = 255
        patch["flat"] = np.array([(x0 + 1.5 * cell) / w, 1.0 - (y0 + 1.5 * cell) / h])
        th = np.linspace(-np.pi - cp["pad"], np.pi + cp["pad"], wpx)
        az_l = azimuth_xy(V[outer[1]], centre)
        o = np.argsort(az_l)
        edge_col = np.stack([np.interp(th, az_l[o], loop_cols[o, k], period=2 * np.pi) for k in range(3)], -1)
        tt = np.linspace(0.0, 1.0, hpx)[:, None, None]
        wg = np.clip(tt / 0.6, 0.0, 1.0)
        wg = wg * wg * (3.0 - 2.0 * wg)
        top = TK.hex_rgb(sc["crown"]) * 255.0 if sc.get("crown") else col
        grad = edge_col[None] * (1 - wg) + top[None, None, :] * wg
        X1, Y1 = gx * cell, gy * cell
        out[Y1:Y1 + hpx, X1:X1 + wpx, :3] = np.clip(grad + 0.5, 0, 255).astype(np.uint8)
        out[Y1:Y1 + hpx, X1:X1 + wpx, 3] = 255
        patch["cap"] = dict(x0=X1, y0=Y1, w=wpx, h=hpx, W=w, H=h, th0=float(th[0]), th1=float(th[-1]))
        return out

    rules = list(tc.get("recolor") or [])
    if edit_on("pupil"):
        rules += list(ed["pupil"].get("rules") or [])
    mats = TK.part_materials(tk, ctx, names, recolor=rules, hooks={str(sk_mat.texture): paint} if sk_mat.texture else None)
    for name, over in (tc.get("material") or {}).items():                    # [head.take.material.<name>] overrides
        if name not in mats:
            raise ValueError(f"[head.take.material.{name}]: not a face material (have {names})")
        mats[name] = override(mats[name], over, f"[head.take.material.{name}]")

    # ---- the face mesh: the imported skin + the cap
    faces = list(skin.faces) + cap_faces
    ring_of = np.zeros(n_all)                                                   # position along the cap: 0 at the boundary .. 1 at the crown
    nr = cap_cfg["rings"] + 1
    for r in range(1, nr):
        ring_of[nV + (r - 1) * nl:nV + r * nl] = r / nr
    ring_of[nV + cap_cfg["rings"] * nl] = 1.0
    cap_uv = [_grad_uv(all_loc[list(f)], 1.0 - ring_of[list(f)], patch["cap"], centre[:2]) for f in cap_faces]
    uv = np.concatenate([skin.uv, np.concatenate(cap_uv, 0)], 0)
    nrm = np.zeros((n_all, 3))
    nrm[:nV] = skin.normals
    cap_n = _normals(all_loc, cap_faces)
    for r in range(1, cap_cfg["rings"] + 1):
        ids = nV + np.arange((r - 1) * nl, r * nl)
        w = max(0.0, 1.0 - r / 3.0)                                             # the first rings blend with the skin's own normals
        nrm[ids] = cap_n[ids] * (1 - w) + skin.normals[outer[1]] * w
    nrm[nV + cap_cfg["rings"] * nl] = cap_n[nV + cap_cfg["rings"] * nl]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    weights = {k: np.concatenate([v, np.zeros(n_all - nV)]) for k, v in skin.weights.items()}
    weights.setdefault(head_bone, np.zeros(n_all))
    weights[head_bone][nV:] = 1.0
    morphs = {k: np.concatenate([v, np.zeros((n_all - nV, 3))]) for k, v in skin.morphs.items()}
    face_mesh = Mesh("face", verts=all_loc + pivot, faces=faces, uv=uv, face_mat=np.zeros(len(faces), int), mats=[skin_name],
                     weights=weights, normals=nrm, morphs=morphs)

    # ---- the closed shell (skin, cap, a fan over every opening); its radial map stands in for the implicit shape
    shell_V = [all_loc]
    shell_T = [t for f in faces for t in _fan(f)]
    base = n_all
    for hl in holes:
        apex, hf = CP.hole_fan(V[hl[1]], centre)
        g = np.concatenate([hl[1], [base]])
        base += 1
        shell_V.append(apex[None])
        shell_T += [tuple(int(g[i]) for i in f) for f in hf]
    shell_V = np.concatenate(shell_V, 0)
    nd = V[neck[1]] - centre                                                      # rays above the neck's own opening must hit the shell
    open_deg = min(150.0, float(np.degrees(np.arccos(np.clip(nd[:, 2] / np.linalg.norm(nd, axis=1), -1, 1)).min())) - 0.5)
    rshape = SH.RadialShape.from_shell(centre, shell_V, shell_T, res_deg=float(tc.get("shell_deg", 2.0)), phi_open_deg=open_deg,
                                       anchors=_anchors(V, centre))

    # ---- ears on the shell
    P, Fe, einfo = HR.ear(rshape, cfg.get("ear"))
    ear_V, ear_F, ear_info, off_e = [], [], {}, 0
    for side, sg in (("L", 1.0), ("R", -1.0)):
        Pm = P * [sg, 1.0, 1.0]
        Fm = Fe if sg > 0 else [tuple(reversed(f)) for f in Fe]
        ear_V.append(Pm)
        ear_F += [tuple(i + off_e for i in f) for f in Fm]
        off_e += len(Pm)
        ear_info[side] = {k: (np.asarray(v) * [sg, 1.0, 1.0]) for k, v in einfo.items()}
    ear_V = np.concatenate(ear_V, 0)
    ears = Mesh("ears", verts=ear_V + pivot, faces=ear_F, uv=np.tile(patch["flat"], (sum(len(f) for f in ear_F), 1)),
                face_mat=np.zeros(len(ear_F), int), mats=[skin_name], weights={head_bone: np.ones(len(ear_V))},
                normals=_normals(ear_V, ear_F))

    # ---- the other face materials, one mesh each
    meshes = [face_mesh]
    extra_mats = []
    vertex_morphs = list(panel_of)
    for name in names:
        if name == skin_name:
            continue
        pm = pieces[name]
        m = Mesh(name, verts=pm.verts, faces=pm.faces, uv=pm.uv, face_mat=np.zeros(len(pm.faces), int), mats=[name],
                 weights=pm.weights, normals=pm.normals, morphs=pm.morphs)
        meshes.append(m)
        if edit_on("fang") and ed["fang"].get("material") == name and mouth:
            fc = ed["fang"]
            white_name, line_name = str(fc.get("white_material", "牙")), str(fc.get("outline_material", "牙線"))
            meshes.append(HE.make_fang(skin, mouth[0][1], vertex_morphs, fc, m, [white_name, line_name]))
            extra_mats += [_glow_material(white_name, "fang", ctx.save_png("white_sphere", TX.white_sphere())),
                           _flat_material(line_name, "fang outline", str(fc.get("outline", "#b9566b")))]
    if edit_on("blush"):
        rest_meshes, rest_mats = _blush_rest(ctx, tk, pieces, skin, mats, ed["blush"], panel_of)
        meshes += rest_meshes
        extra_mats += rest_mats
    meshes.append(ears)

    info = _published(ctx, H, tk, tc, rshape, frame, pivot, nc, nt, ear_info, shell_V, shell_T, holes, V)
    declared = [Morph(m.name, m.panel, m.name_en) for m in tk.morphs if m.kind == "vertex"] + mixed
    return Part("head", meshes=meshes, materials=[mats[n] for n in names] + extra_mats, bones=[], morphs=declared, info=info)


def _mix_morphs(pieces, defs, panel_of):
    """[[head.morphs]]: new vertex morphs mixed from the imported ones (head_edit.mix_morphs), each {name, en, panel,
    mix = {morph: weight}, move = {material: [dx, dy, dz]}}; they join `panel_of` (so a bake compensates them like the
    others). Returns their part.Morph declarations."""
    out = []
    for d in defs:
        name, mix, move = str(d.get("name") or ""), dict(d.get("mix") or {}), dict(d.get("move") or {})
        bad = [k for k in mix if k not in panel_of] + [k for k in move if k not in pieces]
        if not name or not (mix or move) or bad or name in panel_of:
            raise ValueError(f"[[head.morphs]] {name!r}: needs a new name and a mix of the face's vertex morphs and/or "
                             f"moves of its materials (unknown: {bad}; morphs: {sorted(panel_of)}; "
                             f"materials: {sorted(pieces)})")
        HE.mix_morphs(pieces, name, mix, move)
        panel = str(d.get("panel", "other"))
        panel_of[name] = panel
        out.append(Morph(name, panel, str(d.get("en", ""))))
    return out


def _bake(ctx, pieces, cfg, default, panel_of, key):
    """[head.edit.lids] / [head.edit.mouth] {rest, morph}: `rest` of the morph baked into the rest shape, every morph of
    its panel compensated (head_edit.bake_rest)."""
    morph, share = str(cfg.get("morph", default)), float(cfg.get("rest", 0.0))
    if morph not in panel_of:
        raise ValueError(f"[head.edit.{key}] morph {morph!r} is not a vertex morph of the face "
                         f"(have {sorted(panel_of)})")
    if not 0.0 <= share < 1.0:
        raise ValueError(f"[head.edit.{key}] rest = {share}: a share from 0 to below 1")
    if share > 0.0:
        comp = [n for n, p in panel_of.items() if p == panel_of[morph] and n != morph]
        done = HE.bake_rest(pieces, morph, share, comp)
        ctx.log(f"{morph} baked in at {share:g}; compensated: "
                + ", ".join(f"{k} ({'+'.join('LR'[s < 0] for s in v)})" for k, v in done.items()))


def _move(pieces, cfg):
    """[head.edit.move] {material = [dx, dy, dz]}: whole materials moved rigidly at rest (the eye highlights lowered
    under heavier lids); their morph offsets stay, so every morph still moves them as it did."""
    for name, vec in cfg.items():
        if name == "enabled":
            continue
        if name not in pieces:
            raise ValueError(f"[head.edit.move] {name!r}: not a face material (have {sorted(pieces)})")
        pieces[name].verts = pieces[name].verts + np.asarray(vec, float).reshape(1, 3)


def override(mat, over, where):
    """A copy of the part.Material `mat` with the fields in `over` replaced (lists become tuples); an unknown field is
    an error. `[head.take.material.<name>] self_shadow = false`: the material takes no cast shadow (the eyes' whites).
    """
    import dataclasses
    fields = {f.name for f in dataclasses.fields(mat)}
    bad = [k for k in over if k not in fields or k == "name"]
    if bad:
        raise ValueError(f"{where}: {bad} are not material fields (have {sorted(fields - {'name'})})")
    return dataclasses.replace(mat, **{k: tuple(v) if isinstance(v, list) else v for k, v in over.items()})


def decal_weights(verts, faces, behind, feather=0.0, middle=1.0):
    """Per vertex of a decal mesh (a fade copy): 1 inside, fading smoothly to 0 over `feather` metres (along its
    surface) toward its open edges and toward its vertices in `behind` (where it passes behind the skin: a hard edge
    otherwise), times `middle` over the middle third of its width (x), easing back to 1 over the next sixth on each side
    (the nose between the cheeks)."""
    n = len(verts)
    w = np.ones(n)
    if feather > 0.0:
        count = {}
        for f in faces:
            for k in range(len(f)):
                e = tuple(sorted((int(f[k]), int(f[(k + 1) % len(f)]))))
                count[e] = count.get(e, 0) + 1
        edge = {i for e, c in count.items() if c == 1 for i in e} | {int(i) for i in np.nonzero(behind)[0]}
        if edge:
            d, _ = surface_distance(verts, faces, sorted(edge))
            w *= HE.smooth3(np.where(np.isfinite(d), d, feather) / feather)
    if middle != 1.0 and n:
        x = verts[:, 0]
        mid, width = 0.5 * (x.min() + x.max()), max(float(np.ptp(x)), 1e-9)
        t = (np.abs(x - mid) - width / 6.0) / (width / 6.0)
        w *= middle + (1.0 - middle) * HE.smooth3(t)
    return w


def _blush_rest(ctx, tk, pieces, skin, mats, cfg, panel_of):
    """[head.edit.blush] {rest, morph, material, feather, middle}: a faint copy of what the morph (照れ) pushes through
    the skin, shown at rest (head_edit.fade_copy), in a copy of its material `rest` as opaque, without its sphere map
    (an additive sphere brightens the transparent texels round the painted blush into a pale band); faces painted
    transparent all over are left out. With `feather` / `middle` the copy gets its own texture, its alpha times
    `decal_weights` (a soft lower edge where the decal meets the skin; less over the nose). Returns (meshes, materials).
    """
    import dataclasses
    morph, share = str(cfg.get("morph", "照れ")), float(cfg.get("rest", 0.0))
    feather, middle = float(cfg.get("feather", 0.0)), float(cfg.get("middle", 1.0))
    if morph not in panel_of:
        raise ValueError(f"[head.edit.blush] morph {morph!r} is not a vertex morph of the face "
                         f"(have {sorted(panel_of)})")
    if not 0.0 <= share <= 1.0 or not 0.0 <= middle <= 1.0 or feather < 0.0:
        raise ValueError(f"[head.edit.blush] rest = {share}, middle = {middle}: shares from 0 to 1; "
                         f"feather = {feather} >= 0")
    meshes, out = [], []
    if share <= 0.0:
        return meshes, out
    for name, pm in pieces.items():
        tex = tk.materials[name].texture
        if morph not in pm.morphs:
            continue
        cp = HE.fade_copy(pm, morph, None if tex is None else TK._image(tex)[..., 3])
        if cp is None:
            continue
        mat_name = str(cfg.get("material", f"{morph}常")) + ("" if not out else str(len(out) + 1))
        src = mats[name]
        texture = src.texture
        if (feather > 0.0 or middle < 1.0) and texture:
            behind = np.array([(HE.front_y(skin, float(p[0]), float(p[2])) or -1.0) < float(p[1]) + 0.0001
                               for p in cp.verts])
            wv = decal_weights(cp.verts, cp.faces, behind, feather, middle)
            img = np.array(TK._image(ctx.tex_dir / texture), copy=True)
            h, w = img.shape[:2]
            starts = np.cumsum([0] + [len(f) for f in cp.faces])
            polys = [np.stack([cp.uv[starts[k]:starts[k + 1], 0], 1.0 - cp.uv[starts[k]:starts[k + 1], 1]], -1)
                     for k in range(len(cp.faces))]
            R = _raster((h, w), polys, [wv[list(f)][:, None] for f in cp.faces])[..., 0]
            hit = np.isfinite(R)
            img[..., 3] = np.where(hit, np.clip(img[..., 3] * np.where(hit, R, 1.0) + 0.5, 0, 255), img[..., 3])
            texture = ctx.save_png(f"{Path(texture).stem}_rest", img.astype(np.uint8))
        out.append(dataclasses.replace(src, name=mat_name, name_en=f"{src.name_en or name} at rest", texture=texture,
                                       diffuse=tuple(src.diffuse[:3]) + (float(src.diffuse[3]) * share,),
                                       sphere="", sphere_mode="none", alpha_blend=True, edge=False, drop_shadow=False,
                                       self_shadow_map=False))
        meshes.append(Mesh(f"{name}_rest", verts=cp.verts, faces=cp.faces, uv=cp.uv,
                           face_mat=np.zeros(len(cp.faces), int), mats=[mat_name], weights=cp.weights,
                           normals=cp.normals, morphs=cp.morphs))
    return meshes, out


def _glow_material(name, name_en, sphere):
    """Pure white whatever the light (an additive white sphere map and full ambient, as the eye highlights): a small white piece that
    sits in a shadow, such as the fang under the upper lip, stays white."""
    from ..part import Material
    return Material(name, name_en=name_en, diffuse=(0.0, 0.0, 0.0, 1.0), ambient=(1.0, 1.0, 1.0), sphere=sphere, sphere_mode="add", toon="",
                    edge=False, drop_shadow=False, self_shadow=False, self_shadow_map=False)


def _flat_material(name, name_en, hexcol, k=0.9):
    """A flat colour for a generated piece: mmd_tools shades (ambient + diffuse) x light, so the two share `k` of the colour."""
    from ..part import Material
    c = np.array([int(hexcol.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0
    d, a = 0.62 * k * c, 0.38 * k * c
    return Material(name, name_en=name_en, diffuse=(float(d[0]), float(d[1]), float(d[2]), 1.0), ambient=(float(a[0]), float(a[1]), float(a[2])),
                    toon="", edge=False, drop_shadow=False, self_shadow=False, self_shadow_map=False)


def _eye_geometry(tk, ec, holes, V, pivot):
    """Per side ("L"/"R"): the eye opening (the skin hole nearest the iris): its loop, the polygon (k, 2) in (x, z), its centre
    and width, the iris centre and the eye bone's name. V is head-local, pivot the head bone."""
    out = {}
    if ec.get("iris") not in tk.faces or not holes:
        return out
    ip = tk.piece([ec["iris"]])
    for side, key in (("L", "left"), ("R", "right")):
        bone = ec[key]
        if bone not in ip.weights or not tk.has_bone(bone):
            continue
        ic = ip.verts[ip.weights[bone] > 0.5].mean(0)
        hp = min(holes, key=lambda h: np.linalg.norm((V[h[1]] + pivot)[:, [0, 2]].mean(0) - ic[[0, 2]]))
        P = V[hp[1]] + pivot
        out[side] = dict(hole=hp, iris=ic, opening=P[:, [0, 2]], centre=P[:, [0, 2]].mean(0), width=float(np.ptp(P[:, 0])), bone=bone,
                         points=P)
    return out


def _anchors(V_skin, centre):
    """Anchors of the published profile measured on the skin: the front-lowest chin point (y, z)."""
    out = {}
    mid = V_skin[np.abs(V_skin[:, 0]) < 0.003]
    low = mid[mid[:, 1] < centre[1] - 0.04]
    if len(low):
        i = int(np.argmin(low[:, 2]))
        out["chin_point"] = (float(low[i, 1]), float(low[i, 2]))
    return out


def _published(ctx, H, tk, tc, shape, frame, pivot, nc, nt, ear_info, shell_V, shell_T, holes, V):
    """The info dict (head.published) with the eyes, brows and the skin shell taken from the imported head."""
    M = frame.to_model
    guides = (ctx.spec.get("proportions") or {}).get("hair_guides") or {}
    ec = dict(left="左目", right="右目")
    ec.update(tc.get("eyes") or {})
    eye_info, landmarks, lid_margin = {}, {}, {}
    for side, g in _eye_geometry(tk, ec, holes, V, pivot).items():
        hp, P, ic = g["hole"], g["points"], g["iris"]
        E = np.asarray(tk.bone(g["bone"]).head, float)
        near = np.abs(P[:, 0] - ic[0]) < 0.012
        top = P[near][np.argmax(P[near][:, 2])] if near.any() else P[np.argmax(P[:, 2])]
        eye_info[side] = dict(center=E, radius=float(np.linalg.norm(ic - E)), pupil=ic, opening=P[:, [0, 2]], lid_top=top)
        landmarks[f"eye.{side}"] = E
        z = P[:, 2] - P[:, 2].mean()
        lid_margin[side] = dict(upper=hp[1][z >= 0], lower=hp[1][z < 0], inner=int(hp[1][np.argmin(np.abs(P[:, 0]))]),
                                outer=int(hp[1][np.argmax(np.abs(P[:, 0]))]), mesh="face")
    brows = {}
    bb, bm = tc.get("brow_bone"), tc.get("brow_material")
    if bb and bm in tk.faces:
        bp = tk.piece([bm])
        if bb in bp.weights:
            Q = bp.verts[bp.weights[bb] > 0.5] - pivot
            for side, sg in (("L", 1.0), ("R", -1.0)):
                q = Q[sg * Q[:, 0] > 0.005]
                if len(q) < 4:
                    continue
                x = sg * q[:, 0]
                edges = np.linspace(x.min(), x.max() + 1e-9, 9)
                pts = [q[(x >= a) & (x < b)].mean(0)[[0, 2]] for a, b in zip(edges[:-1], edges[1:]) if ((x >= a) & (x < b)).any()]
                brows[side] = np.array(pts)
    info = H.published(shape, frame, {}, brows, ear_info, nc, guides, nt, None,
                       custom=dict(eye_info=eye_info, landmarks=landmarks, skin=(M(shell_V), [tuple(t) for t in shell_T])))
    info["lid_margin"] = lid_margin
    mouth = [h for h in holes if abs((V[h[1]] + pivot)[:, 0].mean()) < 0.01]
    if mouth:
        mp = V[mouth[0][1]] + pivot
        zc = mp[:, 2].mean()
        info["mouth"] = dict(slit_upper=mouth[0][1][mp[:, 2] >= zc], slit_lower=mouth[0][1][mp[:, 2] < zc], centre=mp.mean(0),
                             half_width=float(np.ptp(mp[:, 0]) / 2), mesh="face")
    return info
