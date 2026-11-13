"""Small, smooth edits of an imported face (bpy-free, numpy only): a tilt of the eyes, a softer jaw, a fang, a narrower
highlight, shares of morphs baked into the rest shape, and new morphs mixed from the imported ones.

They keep the vertex morphs of the imported face working: a deformation is applied to the vertices AND to the morph offsets of
the vertices it moves (rotated by the same local rotation), so a blink still closes the lids on each other, and the edits that
move only the rest shape (the jaw) leave the offsets as they are. Landmarks (eye centres and widths, the mouth, the neck) are
found on the imported mesh from its boundary loops; nothing here knows a particular model. All lengths are metres, positions in
model space; pieces are `pmx_take.Piece` objects (edited in place).

Spec (`[head.edit]`, every part switchable with `enabled = false`):
  eyes      {tilt_deg, plateau, reach, pieces, keep_bone}   tilt the outer corners up by tilt_deg: the skin, and the listed pieces
            (sclera, lashes, closed-eye pieces ...) turn about each eye's centre, around the forward axis, by that angle fully
            within `plateau` eye widths of the centre and smoothly less out to `reach` eye widths (zero beyond and at the midline);
            vertices weighted to `keep_bone` (the brows) stay; the iris and pupil are not in `pieces`, so they stay round
  jaw       {amount, ...}   a smooth outward displacement along the normal at the lower cheeks and the jaw line (zero at the
            eyes, the nose, the lips and the chin point, and at the neck and the back of the head)
  fang      {material, side, ...}   see `add_fang`
  highlight {material, scale_x}   narrow the largest highlight shape of each eye
  lids, mouth   {rest, morph}   `bake_rest`: a share of a morph (the blink; a mouth shape) baked into the rest shape,
            every morph of its panel compensated so that it still ends where it ended
  blush     {rest, morph}   `fade_copy`: a decal that a morph pushes through the skin (a blush) shown faintly at rest
"""
import numpy as np


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def smooth3(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------------------------------------------------------ eyes
def eye_field(P, centre, width, side, plateau=0.7, reach=1.3, midline=0.5):
    """Weight 0..1 per point (n, 3) for the eye at `centre` (x, z) of `width`: 1 within `plateau` widths, falling smoothly to 0 at
    `reach` widths, and fading to 0 towards the midline (over the inner `midline` of the distance from the centre to it) so the
    two eyes' fields never overlap; only points on the eye's own side count (`side` +1 = +x)."""
    dx, dz = P[:, 0] - centre[0], P[:, 2] - centre[1]
    d = np.hypot(dx, dz)
    w = smooth((reach * width - d) / max((reach - plateau) * width, 1e-9))
    x = side * P[:, 0]
    mid = smooth3(x / max(midline * abs(centre[0]), 1e-9))
    return w * mid


def rotate_xz(V, centre, angle):
    """Rotate points (n, 3) about the axis through (centre x, *, centre z) parallel to y by per-point `angle` (n,), counter-
    clockwise seen from the front (+x towards +z)."""
    c, s = np.cos(angle), np.sin(angle)
    dx, dz = V[:, 0] - centre[0], V[:, 2] - centre[1]
    out = np.array(V, float, copy=True)
    out[:, 0] = centre[0] + c * dx - s * dz
    out[:, 2] = centre[1] + s * dx + c * dz
    return out


def tilt_piece(piece, centre, width, side, tilt, cfg, brow_bone=None):
    """Turn one piece's vertices, normals and morph offsets about the eye. `tilt` (rad) is the outer-corner-up angle."""
    w = eye_field(piece.verts, centre, width, side, float(cfg.get("plateau", 0.7)), float(cfg.get("reach", 1.3)))
    if brow_bone and brow_bone in piece.weights:
        w = w * (1.0 - np.clip(piece.weights[brow_bone], 0.0, 1.0))
    if not w.any():
        return 0
    ang = side * tilt * w                                                    # the left eye turns +, the right one -
    piece.verts = rotate_xz(piece.verts, centre, ang)
    zero = np.zeros(2)
    piece.normals = rotate_xz(piece.normals, zero, ang)
    for name, off in piece.morphs.items():
        piece.morphs[name] = rotate_xz(off, zero, ang)
    return int((w > 0).sum())


def tilt_eyes(pieces, skin_name, eyes, cfg, brow_bone=None):
    """`eyes`: {side ("L"/"R"): (centre (x, z), width)}. Applies the tilt to the skin and the pieces named in cfg["pieces"]."""
    tilt = np.radians(float(cfg.get("tilt_deg", 4.0)))
    names = [skin_name] + [n for n in cfg.get("pieces", ()) if n != skin_name]
    for side, (centre, width) in eyes.items():
        sg = 1.0 if side == "L" else -1.0
        for n in names:
            if n in pieces:
                tilt_piece(pieces[n], centre, width, sg, tilt, cfg, brow_bone)


# ------------------------------------------------------------------------------------------------------------------- jaw
def _half_width(V, zs, win=0.004):
    """Half width of the skin shell at the heights `zs`: the largest |x| within +-win (smoothed)."""
    hw = np.zeros(len(zs))
    for i, z in enumerate(zs):
        sel = np.abs(V[:, 2] - z) < win
        hw[i] = np.abs(V[sel, 0]).max() if sel.any() else 0.0
    k = np.ones(5) / 5.0
    pad = np.concatenate([[hw[0]] * 2, hw, [hw[-1]] * 2])
    return np.convolve(pad, k, "valid")


def jaw_weights(V, N, eye_bottom, neck_z, centre_y, cfg):
    """Weight 0..1 per skin vertex of the jaw edit: 0 in the centre strip (nose, lips, chin point), 0 up to 1 cm under the eye
    and at the neck, 0 behind the face (smooth), 1 on the lower cheek and the jaw line."""
    zs = np.linspace(V[:, 2].min(), V[:, 2].max(), 160)
    hw = np.interp(V[:, 2], zs, _half_width(V, zs))
    a = np.abs(V[:, 0]) / np.maximum(hw, 1e-6)
    a0, a1 = float(cfg.get("strip", (0.38, 0.70))[0]), float(cfg.get("strip", (0.38, 0.70))[1])
    wx = smooth((a - a0) / (a1 - a0))
    top, depth = float(cfg.get("below_eye", 0.010)), float(cfg.get("ramp_top", 0.020))
    wz_top = smooth((eye_bottom - top - V[:, 2]) / depth)
    wz_bot = smooth((V[:, 2] - (neck_z + float(cfg.get("neck_gap", 0.004)))) / float(cfg.get("ramp_neck", 0.016)))
    wy = smooth((centre_y - float(cfg.get("back", 0.0)) - V[:, 1]) / float(cfg.get("ramp_back", 0.03)))
    return wx * wz_top * wz_bot * wy


def soften_jaw(skin, eye_bottom, neck_z, centre_y, cfg):
    """Move the skin along its normals by cfg["amount"] times the jaw weight (the morph offsets stay as they are)."""
    w = jaw_weights(skin.verts, skin.normals, eye_bottom, neck_z, centre_y, cfg)
    skin.verts = skin.verts + float(cfg.get("amount", 0.002)) * w[:, None] * skin.normals
    return w


# ------------------------------------------------------------------------------------------------------------- highlight
def components(faces, n):
    """Connected components of a face list over n vertices: (label (n,), count); vertices not in a face get their own."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for f in faces:
        r = find(int(f[0]))
        for i in f[1:]:
            parent[find(int(i))] = r
    roots = {}
    lab = np.zeros(n, int)
    for i in range(n):
        lab[i] = roots.setdefault(find(i), len(roots))
    return lab, len(roots)


def narrow_highlight(piece, scale_x):
    """Scale the largest connected shape of the piece on each side of the midline in x about its own centre (the main highlight
    of each eye); the smaller shapes (the small round highlight) stay."""
    V = piece.verts
    lab, n = components(piece.faces, len(V))
    for side in (1.0, -1.0):
        best, area = None, -1.0
        for c in range(n):
            sel = lab == c
            if not sel.any() or side * V[sel, 0].mean() <= 0:
                continue
            ext = np.ptp(V[sel], axis=0)
            a = ext[0] * ext[2]
            if a > area:
                best, area = c, a
        if best is None:
            continue
        sel = lab == best
        cx = 0.5 * (V[sel, 0].min() + V[sel, 0].max())
        V[sel, 0] = cx + (V[sel, 0] - cx) * float(scale_x)
    piece.verts = V


# ---------------------------------------------------------------------------------------------------------------- the fang
def front_y(skin, x, z, y0=-0.3):
    """y of the front-most skin surface at (x, z) seen from the front (a ray from (x, y0, z) along +y), or None when it misses or the
    piece has no faces."""
    if not skin.faces:
        return None
    from .head_shell import ray_first_hit
    tris = np.array([(f[0], f[k], f[k + 1]) for f in skin.faces for k in range(1, len(f) - 1)], int)
    t = ray_first_hit(np.array([x, y0, z], float), np.array([[0.0, 1.0, 0.0]]), skin.verts, tris)[0]
    return float(y0 + t) if np.isfinite(t) else None


def make_fang(skin, mouth_loop_ids, morph_names, cfg, teeth, mats):
    """A small fang as its own mesh: a white triangle (the teeth material) with a slightly larger pink outline triangle 0.12 mm behind it
    (a white tooth on pale skin does not read without it), tilted so the base hides in the upper lip and the tip hangs `peek` below
    the upper lip's lower edge, `over` in FRONT of the skin surface there (the lower lip when the mouth is closed): the classic
    yaeba over the lip.

    skin   the face piece (its faces, vertices and morph offsets are used)
    mouth_loop_ids   indices into skin.verts of the mouth opening's boundary
    morph_names   the vertex morphs to carry
    teeth  the teeth Mesh: the fang takes weights and uv from its nearest vertex
    mats   [white material name, outline material name]

    In every morph the fang rides the lip: the offset of the nearest upper-lip vertex (the donor) is copied rigidly into all its
    corners. cfg["reveal"] = {morph: factor} makes it peek more in those morphs: the tips are pulled down by factor * reveal_down and
    forward by factor * reveal_forward (the base stays hidden, the fang just gets longer), so it reads in the smiles and stays a hint
    at rest. cfg keys (metres): side ("L" = +x), x, width, length (tip to base), peek, over, tuck (how far behind the tip the base
    sits), outline_width, reveal, reveal_down, reveal_forward, name (the mesh)."""
    from ..part import Mesh
    sg = 1.0 if str(cfg.get("side", "L")).upper() == "L" else -1.0
    x0 = sg * abs(float(cfg.get("x", 0.0105)))
    w, ln = float(cfg.get("width", 0.0032)), float(cfg.get("length", 0.0023))
    peek, over, tuck = float(cfg.get("peek", 0.0008)), float(cfg.get("over", 0.0004)), float(cfg.get("tuck", 0.0012))
    ow = float(cfg.get("outline_width", 0.0004))
    down, fwd = float(cfg.get("reveal_down", 0.0010)), float(cfg.get("reveal_forward", 0.0003))
    reveal = {str(k): float(v) for k, v in (cfg.get("reveal") or {}).items()}
    L = skin.verts[mouth_loop_ids]
    upper = np.nonzero(L[:, 2] > L[:, 2].mean())[0]
    q = upper[np.argmin(np.abs(L[upper, 0] - x0))]                          # the upper lip's lower edge near x0
    edge, donor = L[q], int(mouth_loop_ids[q])
    z_tip = float(edge[2] - peek)
    y_surf = front_y(skin, x0, z_tip)
    y_tip = (y_surf if y_surf is not None else float(edge[1])) - over
    white = np.array([[x0 - w / 2, y_tip + tuck, z_tip + ln], [x0 + w / 2, y_tip + tuck, z_tip + ln], [x0, y_tip, z_tip]])
    c = white.mean(0)
    push = white - c
    push[:, 1] = 0.0
    push /= np.maximum(np.linalg.norm(push, axis=1, keepdims=True), 1e-12)
    line = white + ow * push + np.array([0.0, 0.00012, 0.0])                # a touch larger and behind
    V = np.vstack([white, line])
    near = int(np.argmin(np.linalg.norm(np.asarray(teeth.verts, float) - white[2], axis=1)))
    corner_uv, k = None, 0
    for f in teeth.faces:
        for i in f:
            if int(i) == near and corner_uv is None:
                corner_uv = np.asarray(teeth.uv[k], float)
            k += 1
    faces = [(0, 2, 1), (3, 5, 4)]                                          # facing forward (-y)
    weights = {b: np.full(6, float(w_[near])) for b, w_ in teeth.weights.items()}
    morphs = {}
    for name in sorted(set(morph_names) | set(reveal)):
        off = skin.morphs.get(name)
        ride = off[donor] if off is not None else np.zeros(3)
        extra = reveal.get(name, 0.0) * np.array([0.0, -fwd, -down])
        if not (np.abs(ride).max() > 0 or np.abs(extra).max() > 0):
            continue
        arr = np.tile(ride, (6, 1))
        arr[[2, 5]] += extra                                                # only the tips are pulled down
        morphs[name] = arr
    return Mesh(str(cfg.get("name", "fang")), verts=V, faces=faces, uv=np.tile(corner_uv if corner_uv is not None else np.zeros(2), (6, 1)),
                face_mat=np.array([0, 1]), mats=list(mats), weights=weights, normals=np.tile([0.0, -1.0, 0.0], (6, 1)), morphs=morphs)


# ------------------------------------------------------------------------------------------------------- morph bakes
def mix_morphs(pieces, name, mix, move=None):
    """A new vertex morph `name` in every piece: the weighted sum of the piece's offsets of the morphs in `mix` ({morph:
    weight}; a morph a piece does not have counts as zero), plus `move` ({piece name: (dx, dy, dz)}): that whole piece
    moved rigidly (the eye highlights following a lowered lid). A piece none of them moves gets none. Returns the number
    of pieces it is in."""
    n = 0
    for key, p in pieces.items():
        acc = None
        for m, w in mix.items():
            off = p.morphs.get(m)
            if off is not None:
                acc = float(w) * off if acc is None else acc + float(w) * off
        if move and key in move:
            shift = np.broadcast_to(np.asarray(move[key], float), p.verts.shape)
            acc = shift.copy() if acc is None else acc + shift
        if acc is not None and np.abs(acc).max() > 0.0:
            p.morphs[name] = acc
            n += 1
    return n


def compensated_sides(pieces, morph, other, side_min=0.25):
    """The sides (+1 = +x, -1) on which the morph `other` moves the vertices that `morph` moves: by the mean length of
    its offsets on them, every side where that is at least `side_min` of the larger of the two (a wink closes one eye
    and barely touches the other: only its own side counts; a half-closed lid moves both)."""
    tot = {1.0: [0.0, 0], -1.0: [0.0, 0]}
    for p in pieces.values():
        b, o = p.morphs.get(morph), p.morphs.get(other)
        if b is None or o is None:
            continue
        on = np.linalg.norm(b, axis=1) > 1e-9
        for sg in (1.0, -1.0):
            sel = on & (sg * p.verts[:, 0] > 0)
            tot[sg][0] += float(np.linalg.norm(o[sel], axis=1).sum())
            tot[sg][1] += int(sel.sum())
    mean = {sg: (s / n if n else 0.0) for sg, (s, n) in tot.items()}
    top = max(mean.values())
    return [sg for sg in (1.0, -1.0) if top > 0.0 and mean[sg] >= side_min * top]


def bake_rest(pieces, morph, share, compensate, side_min=0.25):
    """Bake `share` (0..1) of the vertex morph `morph` into the rest shape of every piece: its vertices move by share
    times its offsets and the morph keeps the rest of the way ((1 - share) times them), so at full weight it still ends
    where it ended. Every morph in `compensate` gives back that share on each side where it moves the baked vertices
    (`compensated_sides`): it loses share times the baked offsets on that side's vertices (those it did not move as
    well), so it too ends where it ended (a half-closed lid, a smile's closed eyes, a wink's closed eye); on a side it
    leaves alone the new rest stays (a wink's open eye keeps the heavier lid). Returns {morph: sides} of the compensated
    morphs."""
    share = float(share)
    sides = {o: compensated_sides(pieces, morph, o, side_min) for o in compensate if o != morph}
    for p in pieces.values():
        b = p.morphs.get(morph)
        if b is None:
            continue
        side = np.sign(p.verts[:, 0])
        p.verts = p.verts + share * b
        for o, sg in sides.items():
            give = np.where(np.isin(side, sg)[:, None], share * b, 0.0)
            if not (sg and np.abs(give).max() > 0.0):
                continue
            p.morphs[o] = (p.morphs[o] if o in p.morphs else np.zeros_like(b)) - give
        p.morphs[morph] = (1.0 - share) * b
    return {o: sg for o, sg in sides.items() if sg}


def _painted(uv, alpha, samples=4):
    """Whether the texture's alpha (h, w; v from the top) is above zero anywhere on a face with corner uv (k, 2; v up):
    sampled at the corners and on a grid of points inside its fan."""
    h, w = alpha.shape
    t = np.linspace(0.0, 1.0, samples + 1)
    a, b = np.meshgrid(t, t)
    ok = a + b <= 1.0
    a, b = a[ok], b[ok]
    pts = [uv[0] + a[:, None] * (uv[j] - uv[0]) + b[:, None] * (uv[j + 1] - uv[0]) for j in range(1, len(uv) - 1)]
    P = np.concatenate(pts, 0)
    x = np.clip((P[:, 0] * w).astype(int), 0, w - 1)
    y = np.clip(((1.0 - P[:, 1]) * h).astype(int), 0, h - 1)
    return bool((alpha[y, x] > 0).any())


def fade_copy(piece, morph, alpha=None):
    """A copy of the faces of `piece` whose vertices the morph `morph` all moves, placed where that morph takes them: a
    decal that the morph pushes out through the skin (a blush) is seen at rest through the copy, which its own material
    makes faint. The copy carries every other morph of the piece as the original does, and `morph` reversed: at full
    weight it sinks back to where the original rested (behind the skin) while the original comes out, so the morph still
    ends where it ended. With the texture's `alpha` (h, w; v from the top) the faces whose painting is transparent all
    over are left out (a renderer may still shade a transparent face). Returns a `pmx_take.Piece` (vertices renumbered,
    `ids` the original indices) or None when the morph moves no whole (painted) face."""
    from ..pmx_take import Piece
    off = piece.morphs.get(morph)
    if off is None:
        return None
    moved = np.linalg.norm(off, axis=1) > 1e-9
    starts = np.cumsum([0] + [len(f) for f in piece.faces])
    keep = [k for k, f in enumerate(piece.faces) if all(moved[int(i)] for i in f)
            and (alpha is None or _painted(piece.uv[starts[k]:starts[k + 1]], alpha))]
    if not keep:
        return None
    used = np.unique([int(i) for k in keep for i in piece.faces[k]])
    local = np.full(len(piece.verts), -1, int)
    local[used] = np.arange(len(used))
    faces = [tuple(int(local[int(i)]) for i in piece.faces[k]) for k in keep]
    corner = np.concatenate([piece.uv[starts[k]:starts[k + 1]] for k in keep], 0)
    morphs = {n: o[used] for n, o in piece.morphs.items() if n != morph and np.abs(o[used]).max() > 0.0}
    morphs[morph] = -off[used]
    return Piece(list(piece.mats), piece.verts[used] + off[used], faces, corner, piece.normals[used],
                 np.asarray(piece.face_mat)[keep] if len(piece.face_mat) else np.zeros(len(keep), int),
                 {b: w[used] for b, w in piece.weights.items() if np.any(w[used] > 0)}, morphs,
                 np.asarray(piece.ids)[used], np.asarray(piece.vert_uv)[used])
