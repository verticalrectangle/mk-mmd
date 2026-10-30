"""skin_collapse: the skin of the arms and legs against its rest shape, the candy wrapper of a twisted sleeve
(docs/design.md: Checks).

The sample op hands over, on every frame, the vertices of the meshes a cast member's armature deforms whose heaviest
bone is one of a limb segment's (the upper arm: the arm bone and the twist bones under it; the forearm: the elbow and
the wrist's twist bones; the thigh and the shin with their D bones), with every triangle that has such a vertex; the
chains a sim moves (hair, a skirt, a sleeve's chain) belong to no limb, and a mesh a cloth modifier moves is left out.
Each segment is cut by planes square to it, from its joint toward the next (the bones' heads), at SLICES fractions of
its length across SPAN, and on each cut the surface's mean distance from the segment (the cut curve's length-weighted
mean radius) is the size of the cross-section, posed against the same cut at rest. A twist past what the rig's weights
can follow pulls the cut in: linear blend skinning pulls a ring of vertices weighted to bones turned far apart about
the limb toward its axis, and the faces between rings turned against each other neck in, so the sleeve goes flat and
jagged (a cut grows when vertices fly out instead). A bend pinches the skin at the joint itself, which SPAN leaves out:
an MMD rig's shoulder, elbow and knee lose volume at every deep bend (it has no corrective shapes), up to a quarter of
a segment from the joint. Each mesh object is measured on its own, so a sleeve and the arm under it each count.

Calibrated on an MMD model in a jacket over the arms (twist bones splitting 25/50/75 %): ordinary moves change a cut
by at most 7 %, a forearm turned 155-180 degrees (a hand rolled past what an arm can do) by 22-25 %, professional dances
at their most twisted (upper arms 160-170 degrees) by 15-16 %."""
import time

import numpy as np

from . import CheckError, Metric, metric

# segment: (the region's bones (they and their descendants), where it stops, the axis from this joint to that one)
LIMBS = {"upper_arm": (["arm"], ["elbow"], "arm", "elbow"),
         "forearm": (["elbow"], ["wrist"], "elbow", "wrist"),
         "thigh": (["leg", "?leg_d"], ["knee", "?knee_d"], "leg", "knee"),
         "shin": (["knee", "?knee_d"], ["ankle", "?ankle_d"], "knee", "ankle")}
SPAN = (0.3, 0.8)       # the cuts, as fractions of the segment from its joint: the joints' own pinch stays out
SLICES = 11             # cuts across SPAN (every 5 % of the segment)
MIN_CUT = 0.02          # metres of cut curve at rest under which a cut has too little of the mesh to measure
SUB = 5                 # points per cut segment for its mean distance
CHUNK = 128             # frames measured at once


def _list(v):
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def cut_radius(V, T, a, u, L, fracs):
    """The cross-sections of a surface square to a segment: vertices V (F, n, 3), triangles T (m, 3), the segment from
    a (F, 3) along the unit u (F, 3) for L (F,) metres, cut at `fracs` of its length. Returns (r, length) (F, K): the
    cut curve's length-weighted mean distance from the segment's line (NaN where nothing is cut) and its length."""
    used, T = np.unique(T, return_inverse=True)
    T = T.reshape(-1, 3)
    rel = V[:, used] - a[:, None, :]
    z = np.einsum("fni,fi->fn", rel, u)
    perp = rel - z[..., None] * u[:, None, :]
    zt = z[:, T]                                                     # (F, m, 3)
    lo, hi = zt.min(-1), zt.max(-1)
    s = np.linspace(0.0, 1.0, SUB)[:, None]
    F = len(V)
    r_out, l_out = np.full((F, len(fracs)), np.nan), np.zeros((F, len(fracs)))
    for k, f in enumerate(fracs):
        z0 = f * L                                                   # the triangles the plane crosses, frame by frame
        fi, ti = np.nonzero((lo < z0[:, None]) & (hi >= z0[:, None]))  # (a corner on the plane counts as above it)
        if not len(fi):
            continue
        tz = zt[fi, ti] - z0[fi, None]                               # (c, 3) heights of the corners over the plane
        pts = perp[fi[:, None], T[ti]]                               # (c, 3, 3)
        ends, hit = [], []
        for i, j in ((0, 1), (1, 2), (2, 0)):                        # where the plane crosses each edge
            za, zb = tz[:, i], tz[:, j]
            c = (za < 0.0) != (zb < 0.0)
            w = np.where(c, za / np.where(c, za - zb, 1.0), 0.0)[:, None]
            ends.append(pts[:, i] + (pts[:, j] - pts[:, i]) * w)
            hit.append(c)
        hit, ends = np.stack(hit, -1), np.stack(ends, 1)             # (c, 3), (c, 3, 3): a crossed triangle has two
        order = np.argsort(~hit, axis=-1, kind="stable")[:, :2]
        p1 = np.take_along_axis(ends, order[:, 0, None, None], 1)[:, 0]
        p2 = np.take_along_axis(ends, order[:, 1, None, None], 1)[:, 0]
        two = hit.sum(-1) == 2
        seg = np.where(two, np.linalg.norm(p2 - p1, axis=-1), 0.0)
        r = np.linalg.norm(p1[:, None, :] * (1.0 - s) + p2[:, None, :] * s, axis=-1).mean(-1)
        total = np.bincount(fi, weights=seg, minlength=F)
        l_out[:, k] = total
        with np.errstate(invalid="ignore", divide="ignore"):
            r_out[:, k] = np.where(total > 0.0, np.bincount(fi, weights=np.where(two, r, 0.0) * seg, minlength=F) / total,
                                   np.nan)
    return r_out, l_out


def _cut_bones(V, T, bone, a, u, L, f):
    """The heaviest bones (indices) of the vertices of the triangles that the cut at fraction f crosses (one frame)."""
    z = (V - a) @ u
    zt = z[T] - f * L
    crossed = (zt.min(1) < 0.0) & (zt.max(1) >= 0.0)
    return bone[np.unique(T[crossed])]


@metric
class SkinCollapse(Metric):
    name = "skin_collapse"
    doc = (f"The skin of the arms and legs against its rest shape: each limb segment ({', '.join(LIMBS)}) of the meshes "
           f"the armature deforms is cut square to it at {SLICES} places from {SPAN[0]:.0%} to {SPAN[1]:.0%} of its length, "
           "and each cut's mean distance from the segment is compared with the same cut at rest; a twist the rig's "
           "weights cannot follow pulls the sleeve in (a flat, jagged candy wrapper). The chains a sim moves (hair, a "
           "skirt, a sleeve's chain) and meshes a cloth modifier moves are left out; the joints' own pinch at a deep "
           "bend is outside the cuts. Value: the largest change of a cross-section, as a share of its rest size (a loss "
           "when it collapses, a gain when it swells; `kind` in detail).")
    args = {"cast": "cast member (or armature)", "sides": "L R (default both)",
            "limbs": "segments to measure (default " + " ".join(LIMBS) + ")",
            "span": f"[from, to]: the cuts' place as fractions of a segment from its joint (default {list(SPAN)})"}
    default_max = 0.15

    def needs(self, args, ctx, need):
        arm, _ = ctx.cast(args)
        sides = _list(args.get("sides")) or ["L", "R"]
        limbs = _list(args.get("limbs")) or list(LIMBS)
        bad = [x for x in limbs if x not in LIMBS]
        if bad:
            raise CheckError(f"skin_collapse: unknown limbs {bad} (have {list(LIMBS)})")
        span = tuple(float(x) for x in args.get("span", SPAN))
        if len(span) != 2 or not 0.0 <= span[0] < span[1] <= 1.0:
            raise CheckError(f"skin_collapse: span {list(span)} must be two fractions 0 <= from < to <= 1")
        regions = {f"{x}.{s}": {"bones": [f"{b}.{s}" for b in LIMBS[x][0]], "stop": [f"{b}.{s}" for b in LIMBS[x][1]]}
                   for s in sides for x in limbs}
        need.bone(arm, *sorted({f"{b}.{s}" for s in sides for x in limbs for b in LIMBS[x][2:]}))
        return {"arm": arm, "skin": need.skin(arm, regions), "span": span}

    def compute(self, args, ctx, data, frames, st):
        t0 = time.time()
        arm, sk = st["arm"], data.skin(st["skin"], frames)
        fracs = np.linspace(st["span"][0], st["span"][1], SLICES)
        inv = np.linalg.inv(data.arm_world(arm, frames))
        R, P, tri = sk["rest"].astype(float), sk["posed"], sk["triangles"]
        left_out = f"; left out: {sk['skipped']}" if sk["skipped"] else ""
        if not len(tri):
            raise CheckError(f"skin_collapse: no skinned vertex of {arm} belongs to a limb ({', '.join(sk['regions'])})"
                             + left_out)
        worst, per = None, {}
        for g, region in enumerate(sk["regions"]):
            limb, _, side = region.partition(".")
            W, rest, _ = data.bones(arm, [f"{b}.{side}" for b in LIMBS[limb][2:]], frames)
            heads = np.einsum("fij,fkj->fki", inv[:, :3, :3], W[..., :3, 3]) + inv[:, None, :3, 3]   # armature space
            a, d = heads[:, 0], heads[:, 1] - heads[:, 0]
            L = np.linalg.norm(d, axis=1)
            a0, d0 = rest[0, :3, 3], rest[1, :3, 3] - rest[0, :3, 3]
            L0 = np.linalg.norm(d0)
            mine = (sk["region"][tri] == g).any(1)
            for o, obj in enumerate(sk["objects"]):
                T = tri[mine & (sk["object"][tri[:, 0]] == o)]
                if not len(T):
                    continue
                r0, len0 = cut_radius(R[None], T, a0[None], (d0 / L0)[None], np.array([L0]), fracs)
                keep = len0[0] >= MIN_CUT
                if not keep.any():
                    continue
                ratio = np.full((len(frames), len(fracs)), np.nan)
                for c in range(0, len(frames), CHUNK):
                    sl = slice(c, c + CHUNK)
                    r, _ = cut_radius(P[sl].astype(float), T, a[sl], d[sl] / L[sl, None], L[sl], fracs)
                    ratio[sl] = r / r0[0]
                ratio[:, ~keep] = np.nan
                if np.isnan(ratio).all():
                    continue
                change = np.abs(ratio - 1.0)
                fi, ki = np.unravel_index(np.nanargmax(change), change.shape)
                entry = {"loss": round(max(0.0, 1.0 - float(np.nanmin(ratio))), 4),
                         "swell": round(max(0.0, float(np.nanmax(ratio)) - 1.0), 4),
                         "at_frame": int(frames[fi]), "at": round(float(fracs[ki]), 2), "object": obj}
                if region not in per or max(entry["loss"], entry["swell"]) > max(per[region]["loss"], per[region]["swell"]):
                    per[region] = entry
                if worst is None or change[fi, ki] > worst[0]:
                    worst = (float(change[fi, ki]), float(ratio[fi, ki]), fi, ki, region, obj, T, a, d, L)
        if worst is None:
            raise CheckError(f"skin_collapse: no cut of {', '.join(sk['regions'])} crosses {MIN_CUT * 100:g} cm of mesh"
                             + left_out)
        value, ratio, fi, ki, region, obj, T, a, d, L = worst
        cut = _cut_bones(P[fi].astype(float), T, sk["bone"], a[fi], d[fi] / L[fi], L[fi], fracs[ki])
        cut = cut[cut >= 0]
        detail = {"at_frame": int(frames[fi]), "kind": "swell" if ratio > 1.0 else "collapse", "region": region,
                  "object": obj, "bone": sk["bones"][int(np.bincount(cut).argmax())] if len(cut) else None,
                  "at": round(float(fracs[ki]), 2), "regions": per, "frames": len(frames),
                  "seconds": round(time.time() - t0, 2)}
        if sk["skipped"]:
            detail["skipped"] = sk["skipped"]
        return value, detail
