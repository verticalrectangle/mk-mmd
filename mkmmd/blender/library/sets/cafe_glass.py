"""The cafe window glass in Blender: the packed condensation noise image and the rain-on-glass material. The layout
maths (opening, UV map, panes, noise image, splat list) is pure numpy in cafe_maths.py.

Condensation fog = a static 16-bit noise image thresholded by the `fog` parameter (0 clear .. 1 all fogged), generated
once into ~/.cache/mk/cafe/ and packed into the .blend. Rain = splats that land on given times (clip seconds), runners
sliding down in columns, beaded droplets in five sizes and a slow film of water, all in the glass shader. A helper
module: it registers no builder."""
import bpy

from .cafe_maths import (BAR_W, BARS_Y, BARS_Z, FOG_SOFT, FRAME_W, OPEN_H, OPEN_W, OPEN_Y, OPEN_Z, default_ticks,  # noqa: F401
                         ensure_fog_png, pane_rects, splat_drops, uv_from_world)  # noqa: F401
from .cafe_nodes import NG, finish, principled


def fog_image(name):
    """The fog image as a packed Non-Color Blender image (the .blend carries it)."""
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.load(ensure_fog_png(), check_existing=False)
        img.name = name
        img.colorspace_settings.name = "Non-Color"
        img.pack()
    return img


# ================================================================= the glass material
SIN_TH = 0.82                       # sine of the drops' contact angle = surface-normal tilt at the rim
K_FOG = 1.0 + 2.0 * FOG_SOFT        # the fog Map Range edges are 1 - fog*K and K - fog*K
CLEAR_TINT = (0.97, 0.985, 1.0)     # optical tint of clean glass (not a pigment: not a palette colour)


def _dome(g, dvec, r, sinth, feather=0.10):
    """Spherical-cap drop of contact radius r: normal tilt = d * sinth / r inside the rim, none outside."""
    dl = g.vm("LENGTH", dvec)
    inside = g.inv(g.ss(dl, g.mul(r, 1.0 - feather), r))
    return g.vm("SCALE", dvec, scale=g.mul(g.div(sinth, r), inside)), inside


def _beads(g, P, cell, rmin, rmax, dens, weight=None):
    """Beaded droplets: one jittered drop per Voronoi cell (random radius, capped to the cell so no dome is cut), some
    cells empty. -> (tilt, coverage)."""
    vor = g.voronoi(P, 1.0 / cell)
    free = g.voronoi(P, 1.0 / cell, feature="N_SPHERE_RADIUS").outputs["Radius"]
    cr, cg, _ = g.sep(vor.outputs["Color"])
    r = g.mn(g.mad(g.pw(cr, 1.5), rmax - rmin, rmin), g.mul(free, 0.98 * cell))
    keep = g.lt(cg, dens)
    tilt, inside = _dome(g, g.vm("SUBTRACT", P, vor.outputs["Position"]), r, SIN_TH)
    k = keep if weight is None else g.mul(keep, weight)
    return g.vm("SCALE", tilt, scale=k), g.mul(inside, keep)


def _runners(g, P, t, cw, seed, p_exist, sp, rad, trail_k):
    """Drops sliding down in columns of width cw (one per column with probability p_exist), each with a wet trail
    behind it. Heads travel top to bottom and wrap; the speed stutters (stick-slip). -> (tilt, wipe mask)."""
    X, Y, _ = g.sep(P)
    ci = g.fl(g.div(X, cw))
    lx = g.sub(X, g.mul(ci, cw))
    v0, col = g.white("1D", w=g.add(ci, seed))
    r_, g_, b_ = g.sep(col)
    ex = g.lt(v0, p_exist)
    cx = g.mul(g.mad(r_, 0.5, 0.25), cw)
    speed = g.mad(g_, sp[1] - sp[0], sp[0])
    H = OPEN_H + 0.6
    stutter = g.mul(g.m("SINE", g.mad(g.mul(t, 1.9), 1.0, g.mul(b_, 40.0))), 0.22)
    s = g.fr(g.add(g.div(g.mul(g.add(t, stutter), speed), H), b_))
    yh = g.sub(OPEN_H + 0.3, g.mul(s, H))
    rr = g.mad(g.div(v0, p_exist), rad[1] - rad[0], rad[0])
    qx = g.div(g.sub(lx, cx), rr)
    dy = g.sub(Y, yh)
    q = g.xyz(qx, g.div(dy, g.mul(rr, 1.35)), 0.0)
    inside = g.mul(g.inv(g.ss(g.vm("LENGTH", q), 0.88, 1.0)), ex)
    head = g.vm("SCALE", q, scale=g.mul(inside, SIN_TH))
    tw = g.mul(rr, trail_k)
    ax = g.ab(g.sub(lx, cx))
    above = g.ss(dy, 0.0, rr)
    fade = g.mul(g.inv(g.ss(dy, 0.25, 1.9)), ex)
    trail = g.mul(g.mul(g.inv(g.ss(ax, g.mul(tw, 0.35), tw)), above), fade)
    ridge = g.mul(g.mul(g.div(g.sub(lx, cx), tw), 0.30), trail)
    wide = g.mul(g.mul(g.inv(g.ss(ax, tw, g.mul(tw, 2.2))), above), g.mul(g.inv(g.ss(dy, 0.9, 2.2)), ex))
    return g.vm("ADD", head, g.xyz(ridge, 0.0, 0.0)), g.mx(inside, wide)


def _splat(g, P, t, s):
    """One rain splat that lands at clip time s['t']: a star-shaped crown that relaxes into a round drop, waits, then
    runs down leaving a trail; scatters tiny beads around the impact. s holds sockets (see splat_group).
    -> (tilt, wipe, bead boost)."""
    X, Y, _ = g.sep(P)
    pu, pv, r0 = g.mul(s["u"], OPEN_W), g.mul(s["v"], OPEN_H), s["r"]
    dt = g.sub(t, s["t"])
    born = g.gt(dt, 0.0)
    tau = g.mx(dt, 0.0)
    e1 = g.m("EXPONENT", g.mul(tau, -1.0 / 0.13))
    rho = g.mul(g.mad(e1, 1.5, 1.0), r0)
    e2 = g.m("EXPONENT", g.mul(tau, -1.0 / 0.20))
    sinth = g.mad(g.inv(e2), 0.42, 0.40)
    f = g.sat(g.div(g.mx(g.sub(tau, s["dwell"]), 0.0), s["run_t"]))
    run = g.mul(g.mul(f, g.mad(f, -2.0, 3.0)), s["run"])
    cy = g.sub(pv, run)
    dx = g.sub(X, pu)
    dy = g.sub(Y, cy)
    ang = g.m("ARCTAN2", dy, dx)
    wob = g.mul(e1, g.mul(g.m("COSINE", g.mad(ang, 5.0, s["ph"])), 0.32))
    R = g.mul(rho, g.add(wob, 1.0))
    tilt, inside = _dome(g, g.xyz(dx, dy, 0.0), R, sinth)
    # trail between the start point and the head
    tw = g.mul(r0, 0.30)
    ax = g.ab(dx)
    half = g.mul(r0, 0.5)
    span = g.mul(g.ss(dy, 0.0, half), g.inv(g.ss(g.sub(Y, pv), 0.0, half)))
    trail = g.mul(g.inv(g.ss(ax, g.mul(tw, 0.35), tw)), span)
    ridge = g.mul(g.mul(g.div(dx, tw), 0.30), trail)
    wide = g.mul(g.inv(g.ss(ax, tw, g.mul(tw, 2.2))), span)
    tilt = g.vm("SCALE", g.vm("ADD", tilt, g.xyz(ridge, 0.0, 0.0)), scale=born)
    wipe = g.mul(g.mx(inside, wide), born)
    dist = g.vm("LENGTH", g.xyz(g.sub(X, pu), g.sub(Y, pv), 0.0))
    boost = g.mul(g.inv(g.ss(dist, g.mul(r0, 2.2), g.mul(r0, 4.6))), born)
    return tilt, wipe, boost


SPLAT_INPUTS = ("ts", "u", "v", "r", "dwell", "run", "run_t", "ph")


def splat_group(name):
    """The splat shader as a node group (inputs P, t and the splat's own parameters SPLAT_INPUTS; outputs tilt, wipe,
    boost): built once and instanced per splat, which keeps the material small to build."""
    grp = bpy.data.node_groups.get(name)
    if grp is not None:
        bpy.data.node_groups.remove(grp)
    grp = bpy.data.node_groups.new(name, "ShaderNodeTree")
    grp.interface.new_socket("P", in_out="INPUT", socket_type="NodeSocketVector")
    grp.interface.new_socket("t", in_out="INPUT", socket_type="NodeSocketFloat")
    for k in SPLAT_INPUTS:
        grp.interface.new_socket(k, in_out="INPUT", socket_type="NodeSocketFloat")
    grp.interface.new_socket("tilt", in_out="OUTPUT", socket_type="NodeSocketVector")
    grp.interface.new_socket("wipe", in_out="OUTPUT", socket_type="NodeSocketFloat")
    grp.interface.new_socket("boost", in_out="OUTPUT", socket_type="NodeSocketFloat")
    g = NG(grp)
    gi, go = g.node("NodeGroupInput"), g.node("NodeGroupOutput")
    s = {"t": gi.outputs["ts"], **{k: gi.outputs[k] for k in SPLAT_INPUTS if k != "ts"}}
    tilt, wipe, boost = _splat(g, gi.outputs["P"], gi.outputs["t"], s)
    for out_name, sock in (("tilt", tilt), ("wipe", wipe), ("boost", boost)):
        grp.links.new(sock, go.inputs[out_name])
    return grp


def glass_tilt(g, P, t, splats, rng, splat_grp):
    """Tangent-space normal tilt (vector, xy) of all the rain on the glass. -> (tilt, wipe)"""
    params = []
    for (ts, u, v, r) in splats:
        params.append({"ts": ts, "u": u, "v": v, "r": r, "dwell": rng.uniform(0.35, 1.0), "run": rng.uniform(0.10, 0.42),
                       "run_t": rng.uniform(1.3, 3.0), "ph": rng.uniform(0.0, 6.28)})
    tilt = None
    wipe = 0.0
    boost = 0.0
    for s in params:
        n = g.node("ShaderNodeGroup")
        n.node_tree = splat_grp
        g.nt.links.new(P, n.inputs["P"])
        g.nt.links.new(t, n.inputs["t"])
        for k in SPLAT_INPUTS:
            n.inputs[k].default_value = s[k]
        tl, wp, bo = n.outputs["tilt"], n.outputs["wipe"], n.outputs["boost"]
        tilt = tl if tilt is None else g.vm("ADD", tilt, tl)
        wipe = g.mx(wipe, wp)
        boost = g.mx(boost, bo)
    if tilt is None:
        tilt = g.xyz(0.0, 0.0, 0.0)
    for (cw, seed, pe, sp, rad, tk) in ((0.10, 3.0, 0.72, (0.020, 0.060), (0.0042, 0.0065), 0.55),
                                        (0.17, 41.0, 0.55, (0.040, 0.100), (0.0070, 0.0105), 0.45)):
        tl, wp = _runners(g, P, t, cw, seed, pe, sp, rad, tk)
        tilt = g.vm("ADD", tilt, tl)
        wipe = g.mx(wipe, wp)
    # the film of water: slow vertical ripples that bend the whole view a little
    X, Y, _ = g.sep(P)
    film = g.noise(g.xyz(g.mul(X, 9.0), g.add(g.mul(Y, 0.9), g.mul(t, -0.05)), 0.0), 1.0, detail=2.0, rough=0.5)
    tilt = g.vm("ADD", tilt, g.xyz(g.mul(g.sub(film, 0.5), 0.16), 0.0, 0.0))
    dist = g.node("ShaderNodeCameraData").outputs["View Distance"]
    w_tny = g.inv(g.ss(dist, 1.6, 3.6))
    w_sml = g.inv(g.ss(dist, 3.2, 7.0))
    cl = g.mad(g.noise(P, 4.0, detail=2.0, rough=0.5), 1.5, 0.30)
    t_hero, m_hero = _beads(g, P, 0.105, 0.007, 0.016, g.mul(0.20, cl))
    t_big, m_big = _beads(g, P, 0.055, 0.0045, 0.0105, g.mul(0.30, cl))
    t_med, m_med = _beads(g, P, 0.026, 0.0028, 0.0060, g.mul(0.42, cl))
    t_sml, m_sml = _beads(g, P, 0.0115, 0.0012, 0.0030, g.add(g.mul(0.50, cl), g.mul(boost, 0.6)), w_sml)
    t_tny, m_tny = _beads(g, P, 0.0052, 0.0005, 0.0014, g.add(g.mul(0.55, cl), g.mul(boost, 0.8)), w_tny)
    beads, cover = t_hero, m_hero
    for tl, own in ((t_big, m_big), (t_med, m_med), (t_sml, m_sml), (t_tny, m_tny)):
        # each smaller layer is erased wherever a bigger drop already sits
        beads = g.vm("ADD", beads, g.vm("SCALE", tl, scale=g.inv(cover)))
        cover = g.mx(cover, own)
    tilt = g.vm("ADD", tilt, g.vm("SCALE", beads, scale=g.inv(wipe)))
    return tilt, wipe


def mat_glass(name, params, splats, rng, fog_tint, fog_img):
    """The window glass material. Principled glass (refraction by screen-space ray tracing, thin slab) whose normal
    comes from the procedural rain; condensation fog from the noise image. `fog_tint` is the linear colour of the
    condensation."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    for n in list(m.node_tree.nodes):
        m.node_tree.nodes.remove(n)
    m.surface_render_method = "DITHERED"
    g = NG(m.node_tree, params)
    uvn = g.node("ShaderNodeUVMap", uv_map="UVMap")
    u, v, _ = g.sep(uvn.outputs[0])
    P = g.xyz(g.mul(u, OPEN_W), g.mul(v, OPEN_H), 0.0)
    t = g.param("clip_t")
    # ---- fog: image lookup + smoothstep whose edges follow the fog parameter
    tex = g.node("ShaderNodeTexImage", image=fog_img, interpolation="Linear", extension="EXTEND", projection="FLAT")
    g.put(tex.inputs["Vector"], uvn.outputs[0])
    nfog = g.sep(tex.outputs["Color"])[0]
    fk = g.mul(g.param("fog"), K_FOG)
    F = g.ss(nfog, g.sub(1.0, fk), g.sub(K_FOG, fk))
    # ---- rain
    tilt, wipe = glass_tilt(g, P, t, splats, rng, splat_group(f"{name}_splat"))
    tx, ty, _ = g.sep(tilt)
    tz = g.sq(g.mx(g.sub(1.0, g.add(g.mul(tx, tx), g.mul(ty, ty))), 0.04))
    enc = g.rgb(g.mad(tx, 0.5, 0.5), g.mad(ty, 0.5, 0.5), g.mad(tz, 0.5, 0.5))
    nm = g.node("ShaderNodeNormalMap", space="TANGENT", uv_map="UVMap")
    g.put(nm.inputs["Color"], enc)
    # ---- shading, seen from the room (front face): refractive glass, clear -> frosted + whitened by the fog
    base = g.mixc(CLEAR_TINT, fog_tint, F)
    grain = g.noise(g.xyz(g.mul(u, 90.0), g.mul(v, 76.0), 0.0), 1.0, detail=3.0, rough=0.55)
    rough = g.mixf(0.012, g.mad(grain, 0.3, 0.42), F)
    trans = g.mixf(1.0, 0.66, F)
    specl = g.mixf(0.14, 1.0, g.param("refl"))
    front = principled(g, base, rough=rough, spec=specl, normal=nm.outputs[0], transmission_weight=trans, IOR=1.45,
                       emission_color=fog_tint, emission_strength=g.mul(F, 0.10))
    # ---- seen from outside (back face): screen-space refraction turns lit objects behind the pane (the room) into flat
    # grey silhouettes, so here the pane is alpha-transparent: drops are milky glossy beads, fog a translucent veil
    a_drop = g.mul(g.pw(g.sat(g.div(g.vm("LENGTH", tilt), 0.8)), 1.1), 0.6)
    back = principled(g, g.mixc((0.96, 0.98, 1.0), fog_tint, F), rough=g.mixf(0.05, 0.5, F), spec=0.8,
                      normal=nm.outputs[0], alpha=g.mx(a_drop, g.mul(F, 0.80)),
                      emission_color=fog_tint, emission_strength=g.mul(F, 0.10))
    mix = g.node("ShaderNodeMixShader")
    g.put(mix.inputs[0], g.node("ShaderNodeNewGeometry").outputs["Backfacing"])
    g.nt.links.new(front.outputs[0], mix.inputs[1])
    g.nt.links.new(back.outputs[0], mix.inputs[2])
    finish(g, mix.outputs[0], thickness=0.004)
    m.use_raytrace_refraction = True
    m.thickness_mode = "SLAB"
    return m
