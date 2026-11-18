"""Split-toe socks and straw sandals (part `zori`, numpy only).

  socks     her skin from the heel up to `top` over the ankle joints, `snug` off it, relaxed, its top edge a soft roll
            `roll` thick; its first `toes` metres a toe box lofted forward through her foot's convex sections (the
            small toes in one, her feet inside hidden), going over from the sock behind it over `blend` metres and
            rounded at its front over `nose` metres, one cloth with the sock (its texture runs on); between the big
            toe and the others a cleft in the cloth itself, `cleft` deep from the toes' front, `width` wide where it
            opens, at `at` of the toes' width from the inner side (the big toe's), its groove sunk `sink` along the top
  sole      under each foot a sole whose top fills the space between the floor and the sock (the foot rests on it as
            it stands: a wedge where her heel stands higher), its underside flat on the floor (z = 0), its outline the
            foot's footprint `margin` wider all round, rounded into an oval (`round` metres of smoothing): a straw pad
            `layer` thick on top, its edge rounded `edge`, `ridges` soft straw ridges `ridge` deep round its side, and
            under it the rest of the sole drawn in `inset` all round in the straw's shade, its foot rounded `edge`
  strap     a dark V strap `width` wide and `thick` thick from the cleft back over the sock to the sole's two sides,
            `back` of the way from the toes' front to the heel, lying `lift` over the sock

Sizes are metres x S (S = head_tip.z / 1.7 with leg_extra left out: 0.793 on the bases). Spec: `[zori]` (DEFAULTS
below, tables merge key by key), `[colors.socks]` `base`, `shadow` and `[colors.sandals]` `sole`, `sole_shadow`,
`strap`. Each takes the weights of the skin under it (the sole and the strap her foot's: the ankle and the toes), so
they bend with the foot and the leg IK stays as it is. Materials 足袋, 草履 (the straw), 草履台 (the base) and 鼻緒;
`info` holds the measured numbers (the sock's top, the cleft's depth, the sole's thickness under the heel, the ball
and the toes)."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_fit as OF
from . import outfit_geo as G
from . import outfit_shell as SH
from . import outfit_wear as WR
from .gown import ink, shadow_of, tidy, weld_shell
from .hair_tex import toon_ramp
from .sweater import convex_hull

DEFAULTS = {
    "socks": {"top": 0.052, "snug": 0.0022, "roll": 0.0038, "relax": 6, "gap": 0.0012, "toes": 0.07,
              "blend": 0.019, "nose": 0.015, "cleft": 0.0227, "width": 0.009, "at": 0.40, "sink": 0.003},
    "sole": {"margin": 0.0063, "round": 0.012, "layer": 0.0113, "inset": 0.0025, "edge": 0.0038, "ridges": 3,
             "ridge": 0.0016},
    "strap": {"width": 0.0113, "thick": 0.005, "back": 0.42, "lift": 0.0008},
}
COLORS = {"socks": {"base": "#fbf7f2", "shadow": "#d9cfc6"},
          "sandals": {"sole": "#c8ad74", "sole_shadow": "#93794a", "strap": "#4a2a26"}}
MAT_SOCK = "足袋"
MAT_SOLE = "草履"
MAT_BASE = "草履台"
MAT_STRAP = "鼻緒"
SOCK, SOLE, BASE, STRAP = 0, 1, 2, 3
TAU = 2.0 * np.pi


def foot_bones(names, side):
    """The bones of one foot (its ankle and toes) among `names`."""
    jp = "左" if side == "L" else "右"
    return [b for b in names if b.startswith(jp) and ("足首" in b or "つま先" in b or "足先" in b)]


# ---------------------------------------------------------------- the socks
def toe_box(V, side):
    """(x of its inner side, x of its outer side, y of its front) of a foot's toes: the vertices V (n, 3) of that foot
    within 1.5 cm of its front."""
    sg = 1.0 if side == "L" else -1.0
    F = V[(V[:, 0] * sg > 0.0)]
    y_tip = float(F[:, 1].min())
    toes = F[F[:, 1] < y_tip + 0.015]
    ax = np.abs(toes[:, 0])
    return float(ax.min()), float(ax.max()), y_tip


def cleft(P, side, box, sc, S):
    """The sock's points P (n, 3) of one foot with the cleft between its big toe and the others pressed in (module
    docstring: socks): a V notch in the toes' front, closing `cleft` behind it, and a shallow groove along the top."""
    sg = 1.0 if side == "L" else -1.0
    x_in, x_out, y_tip = box
    xc = sg * (x_in + sc["at"] * (x_out - x_in))
    depth, half = sc["cleft"] * S, 0.5 * sc["width"] * S
    P = np.array(P, float)
    mine = P[:, 0] * sg > 0.0
    dx = np.abs(P[:, 0] - xc)
    near = np.clip(1.0 - dx / (2.0 * half), 0.0, 1.0)
    v_shape = near ** 1.5
    ahead = np.clip(1.0 - (P[:, 1] - y_tip) / depth, 0.0, 1.0)
    push = depth * v_shape * ahead * mine
    P[:, 1] += push
    groove = sc["sink"] * S * near ** 2 * np.clip(1.0 - (P[:, 1] - y_tip) / (2.2 * depth), 0.0, 1.0) * mine
    top = P[:, 2] > np.percentile(P[mine, 2], 40) if mine.any() else np.zeros(len(P), bool)
    P[:, 2] -= groove * top
    return P, xc


def toe_loft(fit, P, loop, side, y_tip, snug, blend, nose, n_rings=16):
    """The toe box of one sock (module docstring: socks): rings (K, m, 3) from the cut `loop` (its points P[loop], in
    order) forward through the convex sections of her foot `snug` out, going over from the loop's own shape to the
    sections' over `blend` metres, its last `nose` metres a round nose closing on the toes' front; and the point (3,)
    the nose closes on. The first ring is the loop's own points."""
    sg = 1.0 if side == "L" else -1.0
    ring0 = P[loop]
    y0 = float(ring0[:, 1].mean())
    c0 = ring0.mean(0)
    phi = np.arctan2(ring0[:, 2] - c0[2], ring0[:, 0] - c0[0])
    d2 = np.stack([np.cos(phi), np.sin(phi)], 1)

    def section(y):                              # (centre (2,), points (m, 2) in x-z) of her foot's convex section
        seg = fit.skin.slice(np.array([0.0, y, 0.0]), np.array([0.0, 1.0, 0.0])).reshape(-1, 3)
        seg = seg[(seg[:, 0] * sg > 0.0) & (seg[:, 2] < c0[2] + 0.06)]
        H = convex_hull(seg[:, [0, 2]]) if len(seg) >= 3 else np.zeros((0, 2))
        if len(H) < 3:
            return None
        cen = H.mean(0)
        r = SW_ray_hull(H - cen, d2)
        return (cen, cen[None] + d2 * (r + snug)[:, None]) if np.all(np.isfinite(r) & (r > 0)) else None
    first = section(y0)
    delta = ring0[:, [0, 2]] - first[1] if first is not None else np.zeros((len(ring0), 2))
    y_front = y_tip - snug
    t = 1.0 - np.cos(0.5 * np.pi * np.linspace(0.0, 1.0, n_rings))[1:]       # denser toward the toes' front
    rings, cens = [ring0], [c0[[0, 2]]]
    for y in y0 + (y_front + nose - y0) * t:
        s = section(y)
        if s is None:
            break
        cen, pts = s
        pts = pts + (1.0 - G.smoothstep(np.clip((y0 - y) / max(blend, 1e-9), 0.0, 1.0))) * delta
        rings.append(np.column_stack([pts[:, 0], np.full(len(pts), y), pts[:, 1]]))
        cens.append(cen)
    last, cen, y1 = rings[-1], cens[-1], float(rings[-1][0, 1])
    for a in np.linspace(0.0, 0.5 * np.pi, 7)[1:-1]:          # the nose: a quarter ellipse to the toes' front
        xz = cen[None] + (last[:, [0, 2]] - cen[None]) * np.cos(a)
        rings.append(np.column_stack([xz[:, 0], np.full(len(xz), y1 - (y1 - y_front) * np.sin(a)), xz[:, 1]]))
    return np.array(rings), np.array([cen[0], y_front, cen[1]])


def SW_ray_hull(H, d):
    """(m,) distance along unit rays d (m, 2) from the origin to the convex polygon H (h, 2) round it."""
    from .sweater import ray_hull
    return ray_hull(H, d)


def socks(fit, body_part, c, soup):
    """Both socks into `soup` (material SOCK). Returns (info, the socks' surface (V, T))."""
    S, L = fit.S, fit.L
    sc = c["socks"]
    z_ank = 0.5 * (float(L["ankle.L"][2]) + float(L["ankle.R"][2]))
    z_top = z_ank + sc["top"] * S
    sh = weld_shell(SH.Shell.from_part(body_part))
    sh = sh.clip(0.5 - sh.share(OF.is_arm))
    sh = sh.clip(z_top - sh.verts[:, 2])
    y_tip = float(sh.verts[:, 1].min())
    y_cut = y_tip + sc["toes"] * S
    sh = sh.clip(sh.verts[:, 1] - y_cut)                       # the heel and the ankle from her skin, the toes lofted
    sh = sh.subdivide()
    snug = sc["snug"] * S
    P = sh.offset(snug, relax=int(sc["relax"]), fit=fit, gap=sc["gap"] * S)
    u = SH.limb_angle(sh.verts, L, SH.LEG_CHAIN, ref=(0.0, -1.0, 0.0))
    loops = sh.loops()
    tops = [lp for lp in loops if float(np.mean(sh.verts[lp, 2])) > z_top - 0.01 * S]
    shell = sh.patch(P, u, -sh.verts[:, 2], u_rep=round(0.25 / WR.TILE), v_scale=1.0 / WR.TILE, mat=SOCK,
                     tag="socks", rims=tops, rim_depth=sc["roll"] * S)
    soup.add(shell)
    Vs, Ts = [shell.v], [SH.triangulate(shell.f)]
    info = {"z_top": z_top}
    for lp in loops:
        if float(np.mean(sh.verts[lp, 2])) > z_top - 0.01 * S:
            continue
        side = "L" if float(np.mean(P[lp, 0])) > 0 else "R"
        rings, tip = toe_loft(fit, P, lp, side, y_tip, snug, sc["blend"] * S, sc["nose"] * S)
        box = toe_box(rings.reshape(-1, 3), side)
        flat, xc = cleft(np.concatenate([rings.reshape(-1, 3), tip[None]]), side, box, sc, S)
        K, m = rings.shape[:2]
        # the texture runs on from the sock behind: each line along the toes keeps its loop point's u, v goes on
        # down and forward
        uc = np.unwrap(u[lp] * TAU) / TAU
        turns = np.round(np.sum((np.diff(np.append(u[lp], u[lp][0])) + 0.5) % 1.0 - 0.5))
        uu = np.broadcast_to(np.append(uc, uc[0] + turns) * round(0.25 / WR.TILE), (K, m + 1))
        vv = -(rings[:, :, 2] + (rings[:, :, 1] - rings[0, 0, 1])) / WR.TILE
        rings, tip = flat[:-1].reshape(rings.shape), flat[-1]
        body = G.loft(rings, True, u=uu, v=np.concatenate([vv, vv[:, :1]], 1), mat=SOCK, tag=f"toes_{side}")
        cap = G.fan(tip, rings[-1], mat=SOCK, tag=f"toes_{side}")
        piece = G.join([body, cap])
        piece.mat = SOCK
        piece = G.orient_outward(piece, np.broadcast_to(rings.reshape(-1, 3).mean(0), (1, 3))[0])
        bones = foot_bones(list(fit.skin.weights), side)
        piece.w = fit.weights_for(piece.v, bones, default=next((b for b in bones if "足首" in b), "下半身"),
                                  faces=piece.f, smooth=2)
        soup.add(piece)
        Vs.append(piece.v)
        Ts.append(SH.triangulate(piece.f) + sum(len(v) for v in Vs[:-1]))
        info[f"cleft_{side}"] = {"x": xc, "y_tip": box[2], "depth": sc["cleft"] * S}
    V = np.concatenate(Vs)
    info["min_gap"] = float(fit.skin.closest(P[sh.verts[:, 2] > z_ank])[0].min())
    return info, (V, np.concatenate(Ts))


# ---------------------------------------------------------------- the soles
def outline(points, margin, smooth, n=96):
    """A closed outline (n, 2) round 2d points: their convex hull `margin` out, rounded by `smooth` metres (an oval
    with no corners), resampled evenly."""
    H = convex_hull(points)
    c = H.mean(0)
    th = np.linspace(0.0, TAU, 720, endpoint=False)
    d = np.stack([np.cos(th), np.sin(th)], 1)
    proj = (H - c) @ d.T                                      # support function of the hull
    h0 = proj.max(0)
    h = h0.copy()
    k = max(int(smooth / (TAU * (h.mean() + margin) / len(th))), 1)
    for _ in range(3):                                        # smoothing the support function rounds the corners
        h = np.convolve(np.concatenate([h[-k:], h, h[:k]]), np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k]
    h = h + margin + max(float(np.max(h0 - h)), 0.0)          # and `margin` out of the foot everywhere
    dh = np.gradient(h, th)
    P = c + d * h[:, None] + np.stack([-np.sin(th), np.cos(th)], 1) * dh[:, None]
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    at = np.arange(n) * s[-1] / n
    Pc = np.vstack([P, P[:1]])
    return np.stack([np.interp(at, s, Pc[:, 0]), np.interp(at, s, Pc[:, 1])], 1)


def sole(fit, c, side, sock_P, sock_T):
    """One sole (module docstring: sole): (patches (material SOLE), its outline (n, 2), its top's height over the
    outline's points and numbers)."""
    S = fit.S
    so = c["sole"]
    sg = 1.0 if side == "L" else -1.0
    mine = sock_P[:, 0] * sg > 0.0
    F = sock_P[mine]
    low = F[F[:, 2] < float(F[:, 2].min()) + 0.012 * S]
    ol = outline(low[:, :2], so["margin"] * S, so["round"] * S)
    # the top: what a straight edge laid under the sock from heel to toes touches (the lower convex envelope of the
    # sock's underside along the foot), so the foot stands on it at the heel and the ball, the arch bridged
    ys = np.linspace(float(F[:, 1].min()), float(F[:, 1].max()), 48)
    half = 0.5 * (ys[1] - ys[0]) + 0.002 * S
    zmin = np.array([F[np.abs(F[:, 1] - y) < half, 2].min() if (np.abs(F[:, 1] - y) < half).any() else np.inf
                     for y in ys])
    ok = np.isfinite(zmin)
    H = convex_hull(np.column_stack([ys[ok], zmin[ok]]))
    lower = H[np.argsort(H[:, 0])]
    keep = [lower[0]]
    for p in lower[1:]:                          # the hull's lower chain: from the heel's end to the toes' end
        while len(keep) >= 2 and (keep[-1][0] - keep[-2][0]) * (p[1] - keep[-2][1]) \
                - (keep[-1][1] - keep[-2][1]) * (p[0] - keep[-2][0]) <= 0:
            keep.pop()
        keep.append(p)
    keep = np.array(keep)

    def top_at(xy):
        y = np.atleast_2d(xy)[:, 1]
        return np.interp(y, keep[:, 0], keep[:, 1]) - 0.0004
    cen = ol.mean(0)
    n = len(ol)
    e, layer, inset = so["edge"] * S, so["layer"] * S, so["inset"] * S
    zt = top_at(ol)
    out = G.unit(np.column_stack([ol - cen, np.zeros(n)]))

    def ring(z, dr):
        return np.column_stack([ol, np.broadcast_to(np.asarray(z, float), (n,))]) \
            + out * np.asarray(dr, float).reshape(-1, 1)
    # the straw pad: its top edge rounded, `ridges` soft ridges down its side
    pad = []
    for a in np.linspace(0.0, 0.5 * np.pi, 5):                 # the rounded top edge, a quarter circle
        pad.append(ring(zt - e * (1.0 - np.cos(a)), -e * (1.0 - np.sin(a))))
    for t in np.linspace(0.0, 1.0, 6 * int(so["ridges"]) + 1)[1:]:
        z = zt - e - t * (layer - e)
        pad.append(ring(z, so["ridge"] * S * np.sin(np.pi * so["ridges"] * t) ** 2))
    pad.append(ring(zt - layer, -inset))                       # under the pad, in to the base
    pad = np.array(pad)
    # the base under it, drawn in all round, its foot rounded
    base = []
    zb_top = zt - layer
    for t in np.linspace(0.0, 1.0, 4):
        base.append(ring(zb_top * (1.0 - t) + e * t, -inset))
    for a in np.linspace(0.0, 0.5 * np.pi, 5)[1:]:
        base.append(ring(e * (1.0 - np.sin(a)), -inset - e * (1.0 - np.cos(a))))
    base = np.array(base)

    def loft(R, mat, tag, u_rep):
        p = G.loft(R, True, u=np.linspace(0.0, u_rep, n + 1), v=np.linspace(1.0, 0.0, len(R)), mat=mat, tag=tag)
        ref = np.column_stack([np.broadcast_to(cen, (R.size // 3, 2)), R.reshape(-1, 3)[:, 2]])
        return G.orient_outward(p, ref)
    side_p = loft(pad, SOLE, f"sole_pad_{side}", 8.0)
    base_p = loft(base, BASE, f"sole_base_{side}", 8.0)

    def cap(ring_pts, z_of, up, mat):                           # a flat face: rings shrinking to the middle
        R = []
        for f in np.linspace(1.0, 0.0, 8)[:-1]:
            xy = cen + (ring_pts[:, :2] - cen) * f
            R.append(np.column_stack([xy, z_of(xy)]))
        R = np.array(R)
        p = G.loft(R, True, u=np.linspace(0.0, 1.0, n + 1), v=np.linspace(0.0, 1.0, len(R)), mat=mat)
        apex = np.array([cen[0], cen[1], float(z_of(cen[None])[0])])
        joined = G.join([p, G.fan(apex, R[-1], mat=mat)])
        joined.mat = mat
        return G.orient_outward(joined, joined.v - np.array([0.0, 0.0, 1.0 if up else -1.0]) * 0.01)
    top = cap(pad[0], top_at, True, SOLE)
    top.tag = f"sole_top_{side}"
    top.uv = (top.v[np.concatenate([np.asarray(f) for f in top.f])][:, :2] - cen) / (0.03 * S) * 0.25
    bot = cap(base[-1], lambda xy: np.zeros(len(np.atleast_2d(xy))), False, BASE)
    bot.tag = f"sole_bottom_{side}"
    yy = ol[:, 1]
    nums = {"heel": float(top_at(np.array([[cen[0], yy.max() - 0.02 * S]]))[0]),
            "ball": float(top_at(np.array([[cen[0], yy.min() + 0.45 * np.ptp(yy)]]))[0]),
            "toes": float(top_at(np.array([[cen[0], yy.min() + 0.08 * np.ptp(yy)]]))[0]),
            "width": float(np.ptp(ol[:, 0])), "length": float(np.ptp(ol[:, 1])), "pad": layer}
    return [side_p, base_p, top, bot], ol, top_at, nums


# ---------------------------------------------------------------- the straps
def strap(fit, c, side, sock_surf, ol, top_at, cleft_info):
    """One V strap (module docstring: strap): patches (material STRAP)."""
    S = fit.S
    st = c["strap"]
    xc, y_tip = cleft_info["x"], cleft_info["y_tip"]
    y_back = y_tip + st["back"] * (float(ol[:, 1].max()) - y_tip)
    near = np.abs(ol[:, 1] - y_back)
    inner = ol[np.argmin(near + 10.0 * (np.abs(ol[:, 0]) > abs(xc)))]
    outer = ol[np.argmin(near + 10.0 * (np.abs(ol[:, 0]) < abs(xc)))]
    root = np.array([xc, y_tip + cleft_info["depth"] * 0.85, 0.0])
    root[2] = float(top_at(root[None, :2])[0])
    r = 0.5 * st["width"] * S
    parts = []
    for end in (inner, outer):
        a = np.array([end[0], end[1], float(top_at(end[None])[0])])
        t = np.linspace(0.0, 1.0, 24)
        line = root[None] + (a - root)[None] * t[:, None] + np.array([0.0, 0.0, 0.05 * S]) * np.sin(np.pi * t)[:, None]
        d, q, tri, bary = sock_surf.closest(line)
        n = sock_surf.outward(tri, bary)
        path = q + n * (st["lift"] * S + 0.5 * st["thick"] * S)
        path[0] = root + np.array([0.0, 0.0, 0.5 * st["thick"] * S])
        path[-1] = a + np.array([0.0, 0.0, 0.5 * st["thick"] * S])
        for _ in range(3):                                     # a smooth line lying on the sock
            path[1:-1] = 0.5 * path[1:-1] + 0.25 * (path[:-2] + path[2:])
            d, q, tri, bary = sock_surf.closest(path[1:-1])
            n = sock_surf.outward(tri, bary)
            low = d < st["lift"] * S + 0.5 * st["thick"] * S
            fix = q[low] + n[low] * (st["lift"] * S + 0.5 * st["thick"] * S)
            path[1:-1][low] = fix
        radii = np.column_stack([np.full(len(path), 0.5 * st["thick"] * S), np.full(len(path), r)])
        parts += G.tube(path, radii, n=10, caps=(True, True), mat=STRAP, tag=f"strap_{side}")
    return parts


# ---------------------------------------------------------------- the part
@builder("zori", needs=("body",))
def build(ctx):
    c = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = {"socks": WR.colours(ctx, "socks", COLORS["socks"]), "sandals": WR.colours(ctx, "sandals",
                                                                                     COLORS["sandals"])}
    S = fit.S
    body_part = ctx.need("body")
    soup = G.Soup("zori", [MAT_SOCK, MAT_SOLE, MAT_BASE, MAT_STRAP])
    info, (P, T) = socks(fit, body_part, c, soup)
    sock_surf = OF.Skin(P, T)
    sock_surf._flip = False
    names = list(fit.skin.weights)
    for side in ("L", "R"):
        bones = foot_bones(names, side)
        ankle = next((b for b in bones if "足首" in b), bones[0] if bones else "下半身")
        pieces, ol, top_at, nums = sole(fit, c, side, P, T)
        for p in pieces:
            p.w = fit.weights_for(p.v, bones, default=ankle)
            soup.add(p)
        for p in strap(fit, c, side, sock_surf, ol, top_at, info[f"cleft_{side}"]):
            p.w = fit.weights_for(p.v, bones, default=ankle)
            soup.add(p)
        info[f"sole_{side}"] = nums
    mesh = tidy(soup.mesh())
    mats = materials(ctx, col)
    ctx.log(f"socks to {info['z_top']:.3f} m, soles {info['sole_L']['heel']:.3f} m under the heel, "
            f"{info['sole_L']['ball']:.3f} m under the ball, {info['sole_L']['toes']:.3f} m under the toes")
    return Part("zori", meshes=[mesh], materials=[m for m in mats if m.name in mesh.mats], info=info)


def straw(rgb, shade, size=128, lines=14):
    """A tile of woven straw: soft ridges along it (v) and a faint weave across, two shades of the straw."""
    u = (np.arange(size) + 0.5) / size
    U, V = np.meshgrid(u, u)
    ridge = 0.5 + 0.5 * np.cos(TAU * lines * U)
    weave = 0.5 + 0.5 * np.cos(TAU * 4 * V + np.pi * (np.floor(lines * U) % 2))
    k = (0.18 * ridge ** 2 + 0.06 * weave)[..., None]
    rgb = np.asarray(rgb, float)[None, None, :] * (1.0 - k) + np.asarray(shade, float)[None, None, :] * k
    return WR._rgba(rgb)


def materials(ctx, col):
    """The socks', soles' and straps' materials: matte, a soft toon step, outlined in a dark shade of the shadow."""
    so, sa = col["socks"], col["sandals"]
    shade = np.asarray(sa["sole_shadow"], float)
    specs = ((MAT_SOCK, "tabi", so["base"], so["shadow"], WR.twill(so["base"], lines=16, depth=0.015), 0.5),
             (MAT_SOLE, "zori", sa["sole"], sa["sole_shadow"], straw(sa["sole"], sa["sole_shadow"]), 0.5),
             (MAT_BASE, "zori_base", shade, shade * 0.8, straw(shade, shade * 0.85, lines=10), 0.5),
             (MAT_STRAP, "hanao", sa["strap"], np.asarray(sa["strap"], float) * 0.7, None, 0.4))
    out = []
    for name, en, base, shade, tex, edge in specs:
        sh = shadow_of({"c": base, "c_shadow": np.asarray(shade, float)}, "c")
        m = WR.material(ctx, name, en, base, tex, shadow=sh, edge_size=edge)
        m.toon = ctx.save_png(f"{en}_toon", toon_ramp(shadow=sh, edge=0.72, soft=0.18))
        m.edge_color = ink(np.asarray(shade, float))
        out.append(m)
    return out
