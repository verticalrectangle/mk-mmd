"""Rin's twin braids with their bows and tufts (bpy-free, numpy only): `build_braids(ctx, fit, cfg, rig, pal) -> Piece`.

Each side is: gather clumps from the scalp behind and below the ear converging onto a black ribbon bow beside the jaw, a
three-strand plait (a chain of slanted lobes on a slim core) hanging over the shoulder and down the side of the chest, a
second bow at the bust line and a small loose tuft below it. One bone chain per side (三つ編左1.., 三つ編右1..) hangs from the
head from the top bow to the tuft tip; every vertex is weighted along it with `Chain.weights` (lobes by their own
position, bows rigidly by the bone they tie, the clumps fading from the head into the first bone). The bow tails hang from
2-bone ribbon chains (リボン左上1,2 / リボン左下1,2) branching off the braid bone at their bow (`bow_tails = "static"` keeps
them rigid).

Config `[hair.braids]` (all lengths in metres, tuned for a head 0.221 m from skull top to chin point and scaled with it, see
DEFAULTS), geometry in hair_braids_path / _plait / _bow, textures in hair_braids_tex."""
import numpy as np

from . import hair_braids_bow as BOW
from . import hair_braids_plait as PL
from . import hair_braids_tex as TX
from . import hair_tex
from .hair_braids_geo import TriMesh, collider_clearance, colliders
from .hair_braids_path import head_refs, make_constraints, plan_side
from .hair_fit import Volume
from .hair_geo import MeshAccum, Piece
from ..part import Material

DEFAULTS = {
    "enabled": True,
    "head_ref": 0.221,            # skull height (top to chin point) the metric defaults are tuned for; sizes scale with it
    # ---- layout
    "bow_dx": 0.0140,             # top bow: outwards of the ear lobe
    "bow_dy": -0.0080,            # ... forwards (-) / backwards (+) of the ear lobe
    "bow_drop": 0.0100,           # ... below the chin point
    "jitter": 0.0012,             # random shift of a bow (m), per side
    "length": None,               # path length from bow knot to bow knot; None = down to the bust line
    "bottom_dz": 0.004,           # bottom bow knot height above the bust line (when length is None)
    "drape_deg": 40.0,            # guide curve: slope from the vertical at the top bow (forward over the shoulder) ...
    "hang_deg": 8.0,              # ... relaxing to this lean further down
    "drape_len": 0.05,            # ... over this length
    "heading_out": 0.25,          # the drape also drifts outwards by this share of its forward travel
    "face_yaw": 20.0,             # the plait's face looks forward, turned outwards by this much (deg)
    "clearance": 0.008,           # centreline to body mesh / skin: this + the braid's mean half size
    "tuft_from_knot": 0.052,      # tuft tip below the bottom bow knot
    # ---- plait: three strands woven over / under
    "width": 0.034,               # braid width (m)
    "thick": 0.024,               # braid thickness
    "taper": 0.10,                # width lost from the top bow to the bottom one
    "lobes": 6,                   # strand crossings between the bows (each one a lobe of the zigzag silhouette)
    "strand_wid": 0.46,           # strand width / braid width
    "strand_thick": 0.44,         # strand thickness / braid thickness
    "weave_depth": 0.30,          # depth amplitude of a strand / braid thickness (crossing strands stay a thickness apart)
    "weave_p": 1.3,               # sharpness of the lateral swing (1 = sine, > 1 pointier edge lobes)
    "weave_phase": 0.0, "weave_dz": 0.0,
    "weave_run_in": 0.004,        # the strands run this far into the bands
    "weave_step": 0.006, "strand_k": 5,
    "pinch": 0.28,                # the plait is this much narrower at a bow's band ...
    "pinch_len": 0.032,           # ... and swells to full width over this length
    # ---- bows
    "top_yaw": 30.0, "bottom_yaw": 8.0,          # how far each bow faces round to the outside of the plait (deg)
    "top_roll": 9.0, "bottom_roll": 5.0,         # wing tilt about the facing axis (deg, mirrored on the other side)
    "top_size": 1.0, "bottom_size": 0.88,        # bow scale
    "wing_len": 0.030, "wing_h": 0.0145, "wing_th": 0.0085,
    "knot_t": 0.0105, "knot_w": 0.0085, "knot_d": 0.0072,
    "band_h": 0.0110, "band_thick": 0.0026,
    "tail_len": 0.031, "tail_width": 0.0095, "tail_notch": 0.0085, "tail_spread": 13.0,
    "bow_tails": "dynamic",       # dynamic: 2-bone ribbon chains | static: rigid on the braid bone
    # ---- tuft
    "tuft_locks": 4, "tuft_len": 0.046, "tuft_width": 0.0125, "tuft_thick": 0.0045, "tuft_fan": 17.0, "tuft_curl": 0.10,
    "tuft_k": 3,
    # ---- gather clumps
    "gather": 5, "gather_theta": (97.0, 125.0), "gather_phi": (72.0, 118.0), "gather_delta": 0.005,
    "gather_leave": 0.045, "gather_drift": 6.0, "gather_spread": 0.020, "gather_width": 0.026, "gather_end_width": 0.013,
    "gather_thick": 0.008, "gather_ring": 0.014, "gather_k": 4,
    # ---- rig
    "bones": 10, "tuft_bones": 2, "radius": 0.011, "ribbon_radius": 0.005, "hold": 0.18,
    "volume": {"top": 0.028, "front": 0.020, "side": 0.034, "back": 0.027},
    # ---- look
    "edge_size": 0.8, "ribbon_edge_size": 0.7, "ribbon": None,
}

SIDE = {1.0: "左", -1.0: "右"}
SIDE_KEY = {1.0: "L", -1.0: "R"}


def merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


class Textures:
    """The PNGs and materials of the braids."""

    def __init__(self, ctx, cfg, pal):
        cb = ctx.get("colors.black", {}) or {}                       # ribbon / ribbon_shade / ribbon_sheen
        colors = {"base": cb.get("ribbon", TX.RIBBON["base"]), "shade": cb.get("ribbon_shade", TX.RIBBON["shade"]),
                  "sheen": cb.get("ribbon_sheen", TX.RIBBON["sheen"])}
        if cfg.get("ribbon"):
            colors["base"] = cfg["ribbon"]
        rng = ctx.rng_for("hair.braids.tex")
        weave = dict(K=float(cfg["lobes"]), phase=float(cfg["weave_phase"]), dz=float(cfg["weave_dz"]))
        self.hair_png = ctx.save_png("braid_atlas", TX.braid_atlas(pal, rng, weave))
        self.ribbon_png = ctx.save_png("braid_ribbon", TX.ribbon_texture(colors, rng=rng))
        hair_colors = ctx.get("colors.hair", {}) or {}
        toon_hair = ctx.save_png("toon_warm", hair_tex.toon_ramp(hair_tex.toon_multiplier(hair_colors)))
        toon_rib = ctx.save_png("toon_ribbon", hair_tex.toon_ramp((0.55, 0.55, 0.64)))
        sphere = ""
        if hasattr(hair_tex, "sphere_ring"):
            sphere = ctx.save_png("sphere_ring", hair_tex.sphere_ring(pal))
        edge_hair = tuple(float(x) for x in np.clip(pal["shadow"] * 0.55, 0, 1)) + (1.0,)
        edge_rib = tuple(float(x) for x in np.clip(hair_tex.srgb(colors["shade"]) * 0.8, 0, 1)) + (1.0,)
        self.hair = Material("三つ編", name_en="braid", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.0, 0.0, 0.0), shininess=0.0,
                             ambient=(0.5, 0.5, 0.5), texture=self.hair_png, toon=toon_hair, sphere=sphere,
                             sphere_mode="add" if sphere else "none", edge=True, edge_color=edge_hair,
                             edge_size=cfg["edge_size"])
        self.ribbon = Material("髪リボン", name_en="hair ribbon", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.26, 0.25, 0.31),
                               shininess=24.0, ambient=(0.45, 0.45, 0.5), texture=self.ribbon_png, toon=toon_rib, edge=True,
                               edge_color=edge_rib, edge_size=cfg["ribbon_edge_size"])


def _vol(fit, cfg):
    v = getattr(fit, "vol", None)
    if v is not None:
        return v
    c = cfg["volume"]
    return Volume(fit.skull, c["top"], c["front"], c["side"], c["back"])



def build_side(ctx, fit, cfg, refs, rig, probes, tex, sx, vol, rng, scale):
    side = SIDE_KEY[sx]
    jp = SIDE[sx]
    body_tm, skin_tm, cols = probes
    need = cfg["clearance"] + 0.25 * (cfg["width"] + cfg["thick"]) * scale
    need_col = cfg["radius"] * scale + 0.006

    def cons(P):
        near = [c for c in cols if (collider_clearance(P, [c]) < 0.2).any()]
        return make_constraints(body_tm, skin_tm, near, need, need_col)

    plan = plan_side(cfg, refs, sx, cons, rng)
    Lb = plan.s_bow[1]
    n_bones, n_tuft = int(cfg["bones"]), int(cfg["tuft_bones"])
    n_main = n_bones - n_tuft
    l = Lb / (n_main - 1)
    s_chain = -0.5 * l + l * np.arange(n_main + 1)                       # bone boundaries down to under the bottom bow
    s_tuft = np.linspace(s_chain[-1], plan.s_end, n_tuft + 1)[1:]
    sig = np.concatenate([s_chain, s_tuft])
    pts = plan.at(sig)[0]
    head_body = ctx.find_body("頭")
    chain = rig.chain(f"三つ編{jp}", pts, "頭", kind="braid", radius=cfg["radius"] * scale, anchor_body=head_body.name if head_body
                      else None, hold=cfg["hold"], name_en=f"braid_{side}_")
    bones = list(chain.names)
    bone_top, bone_bot = bones[0], bones[n_main - 1]

    def wts(P):
        return chain.weights(P)

    blocks = []                                                            # (Block, weights dict (bone -> (n,)), material index)

    def add(block, weights, mat):
        blocks.append((block, weights, mat))

    def bow_weights(sigma, n):
        w = chain.weights(s=np.array([sigma - sig[0]]))
        return {b: np.full(n, float(v[0])) for b, v in w.items()}

    # ---- gather clumps
    items = PL.gather_paths(plan, fit, vol, cfg, scale, rng, 0.5 * l)
    for blk in PL.gather_blocks(plan, items, fit, cfg, scale, rng, TX.strand_uv):
        t = blk.tag["t"]
        sgm = np.clip((t - 0.55) / 0.45, 0.0, 1.0)
        sgm = sgm * sgm * (3.0 - 2.0 * sgm) * 0.40 * l                     # virtual arc coordinate: head -> first bone
        add(blk, chain.weights(s=sgm), 0)
    # ---- the plait: three woven strands between the two bands
    hb = 0.5 * cfg["band_h"] * scale
    W = cfg["width"] * scale
    for blk in PL.weave_blocks(PL.weave_strands(plan, cfg, scale, rng, hb), cfg, TX.plait_uv, rng):
        add(blk, wts(blk.verts), 0)
    # ---- tuft
    for blk in PL.tuft_blocks(plan, cfg, scale, rng, TX.strand_uv):
        add(blk, wts(blk.verts), 0)
    # ---- bows
    bow_info = {}
    for which, sigma, bone in (("top", 0.0, bone_top), ("bottom", Lb, bone_bot)):
        k = cfg[which + "_size"] * scale
        yaw, roll = cfg[which + "_yaw"], cfg[which + "_roll"] * sx * (1.0 if which == "top" else -1.0)
        f_band = float(PL.envelope(cfg, scale, plan, sigma, hb))
        O, t, n, w, t_roll = BOW.bow_frame(plan, sigma, yaw, roll)
        _, T, Bv, N = plan.at(sigma)
        a_e = 0.5 * W * f_band + 0.0030 * scale
        b_e = 0.5 * cfg["thick"] * scale * (2.0 * cfg["weave_depth"] + cfg["strand_thick"]) * f_band + 0.0030 * scale
        psi = np.arctan2(np.dot(n, Bv), np.dot(n, N))
        rho = BOW.ellipse_radius(a_e, b_e, psi)
        wt = bow_weights(sigma, 1)
        bw = lambda cnt: {b: np.full(cnt, float(v[0])) for b, v in wt.items()}
        band = BOW.band_block(O, T, Bv, N, a_e, b_e, cfg["band_h"] * scale * (0.9 if which == "bottom" else 1.0),
                              cfg["band_thick"] * scale)
        add(band, bw(len(band.verts)), 1)
        kc = O + n * (rho + 0.2 * cfg["knot_d"] * k)
        knot = BOW.knot_block(kc, T, n, cfg["knot_t"] * k, cfg["knot_w"] * k, cfg["knot_d"] * k)
        add(knot, bw(len(knot.verts)), 1)
        for ws in (1.0, -1.0):
            wing = BOW.wing_block(kc + ws * 0.6 * cfg["knot_w"] * k * w, ws * w, t_roll, n, cfg["wing_len"] * k,
                                  cfg["wing_h"] * k, cfg["wing_th"] * k)
            add(wing, bw(len(wing.verts)), 1)
        # tails: two ribbons lying on the plait's face below the knot, spreading a little, notched at the ends
        tails = []
        ang = np.radians(cfg["tail_spread"])
        for ws in (1.0, -1.0):
            ds = np.linspace(0.0, 1.0, 5)
            Ls = cfg["tail_len"] * k * (1.0 if ws > 0 else 0.92)
            path = []
            for d in ds:
                sg = sigma + 0.002 + d * Ls
                Pq, Tq, Bq, Nq = plan.at(sg)
                lat = ws * (0.0035 * k + np.tan(ang) * d * Ls * (0.35 + 0.65 * d))
                path.append(Pq + Bq * lat * 1.0 + n * (rho + 0.0016 + 0.0016 * d) * 1.0)
            tails.append(np.array(path))
        tail_blocks = [BOW.tail_block(p, n, cfg["tail_width"] * k, 0.0021 * k, cfg["tail_notch"] * k) for p in tails]
        rb = []
        if cfg["bow_tails"] == "dynamic":
            mid = 0.5 * (tails[0] + tails[1])
            rpts = np.array([mid[0], mid[2], mid[4]])
            names = [f"リボン{jp}{'上' if which == 'top' else '下'}{i + 1}" for i in range(2)]
            body_a = next((b for b in rig.bodies if b.name == bone), None)
            rc = rig.chain(f"リボン{jp}{'上' if which == 'top' else '下'}", rpts, bone, kind="ribbon",
                           radius=cfg["ribbon_radius"] * scale, names=names, anchor_body=body_a.name if body_a else None,
                           name_en=f"ribbon_{side}_{which}_")
            rb = list(rc.names)
            for blk in tail_blocks:
                wr = rc.weights(blk.verts)
                wa = wr.pop(bone)
                bwb = {b: float(v[0]) for b, v in wt.items()}
                for b2, v2 in bwb.items():
                    wr[b2] = wr.get(b2, 0.0) + wa * v2
                add(blk, wr, 1)
        else:
            for blk in tail_blocks:
                add(blk, bw(len(blk.verts)), 1)
        bow_info[which] = {"center": O.copy(), "knot": kc.copy(), "t": t.copy(), "n": n.copy(), "w": w.copy(),
                           "span": 2.0 * cfg["wing_len"] * k, "bone": bone, "ribbon_bones": rb,
                           "tails": [p.copy() for p in tails]}
    # ---- assemble
    acc = MeshAccum(f"hair_braid_{side}", [tex.hair.name, tex.ribbon.name])
    for blk, w, mat in blocks:
        acc.add(blk.verts, blk.faces, blk.uv, mat, weights=w)
    mesh = acc.to_mesh()
    info = {"side": side, "bones": bones, "bow_bones": (bone_top, bone_bot), "chain_pts": chain.pts.copy(), "bows": bow_info,
            "length": float(Lb), "tuft_tip": plan.at(plan.s_end)[0].copy(), "centerline": plan.P[plan.sig >= 0.0].copy(),
            "gather": [{"theta": it["theta"], "phi": it["phi"], "root": it["root"], "dir": it["dir"], "width": it["width"]}
                       for it in items],
            "bounds": (mesh.verts.min(0).copy(), mesh.verts.max(0).copy()), "lobes": int(cfg["lobes"]),
            "ribbon_bones": {k: v["ribbon_bones"] for k, v in bow_info.items()}}
    return mesh, info, plan


def build_braids(ctx, fit, cfg, rig, pal):
    """The twin braids -> Piece (meshes hair_braid_L / hair_braid_R, materials 三つ編 and 髪リボン); bones, bodies and joints go
    into `rig`."""
    cfg = merge(DEFAULTS, cfg)
    if not cfg["enabled"]:
        return Piece(info={"enabled": False})
    body = ctx.parts["body"]
    bm = body.meshes[0]
    sk = ctx.parts["head"].info["skin"]
    probes = (TriMesh(bm.verts, bm.faces), TriMesh(sk["verts"], sk["faces"]), colliders(body))
    refs = head_refs(ctx, fit, cfg)
    scale = refs["scale"]
    vol = _vol(fit, cfg)
    tex = Textures(ctx, cfg, pal)
    meshes, infos, plans = [], {}, {}
    for sx in (1.0, -1.0):
        rng = ctx.rng_for("hair.braids." + SIDE_KEY[sx])
        mesh, info, plan = build_side(ctx, fit, cfg, refs, rig, probes, tex, sx, vol, rng, scale)
        meshes.append(mesh)
        infos[SIDE_KEY[sx]] = info
        plans[SIDE_KEY[sx]] = plan
    frames = {"三つ編": infos["L"]["bones"] + infos["R"]["bones"]}
    rib = [b for s in "LR" for v in infos[s]["ribbon_bones"].values() for b in v]
    if rib:
        frames["髪リボン"] = rib
    info = {"enabled": True, "L": infos["L"], "R": infos["R"], "scale": scale, "width": cfg["width"] * scale,
            "gather": {s: infos[s]["gather"] for s in "LR"}, "bows": {s: infos[s]["bows"] for s in "LR"},
            "bones": {s: infos[s]["bones"] for s in "LR"}, "chain_pts": {s: infos[s]["chain_pts"] for s in "LR"},
            "bounds": {s: infos[s]["bounds"] for s in "LR"}}
    return Piece(meshes=meshes, materials=[tex.hair, tex.ribbon], info=info, frames=frames)
