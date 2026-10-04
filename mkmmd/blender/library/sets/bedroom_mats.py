"""Materials of the bedroom_80s set (procedural shader nodes for EEVEE Next; every colour is a palette-slot blend from
`bedroom_colors.roles`). A helper module: it registers no builder. Detail is put where a camera at 0.5 - 3 m sees it:
paper grain and stripes on the walls, plank grooves, grain and varnish on the parquet, brush marks on the paint."""
import bpy

from ..props.cafe_kit import drive
from .bedroom_maths import PARAMS
from .cafe_nodes import finish, principled


def params_group(name, root):
    """Node group "<name>_params" with one float output per parameter (bedroom_maths.PARAMS); a driver copies each from
    the root's custom property of the same name, so keying the property animates every material that uses the group."""
    g = bpy.data.node_groups.get(f"{name}_params")
    if g is not None:
        bpy.data.node_groups.remove(g)
    g = bpy.data.node_groups.new(f"{name}_params", "ShaderNodeTree")
    out = g.nodes.new("NodeGroupOutput")
    out.location = (300, 0)
    for i, p in enumerate(PARAMS):
        g.interface.new_socket(p, in_out="OUTPUT", socket_type="NodeSocketFloat")
        v = g.nodes.new("ShaderNodeValue")
        v.name = v.label = p
        v.location = (0, -60 * i)
        g.links.new(v.outputs[0], out.inputs[p])
        drive(g, f'nodes["{p}"].outputs[0].default_value', "p", var=("p", root, f'["{p}"]'))
    return g


# ---------------------------------------------------------------------------------------------------- walls, ceiling


def mat_wallpaper(R, L, axis, rail_z, name):
    """Striped 80s wallpaper: a wide dusty stripe, a narrow lighter one with small diamonds, pin stripes between them;
    paper grain; plain above the picture rail. `axis` is the world axis the stripes advance along ("x" or "y")."""
    m, g = R.mat(name)
    x, y, z = g.sep(g.texco("Object"))
    h = x if axis == "x" else y
    period = 0.16
    s = g.fr(g.add(g.mul(h, 1.0 / period), 0.37))
    lite = g.mul(g.ss(s, 0.55, 0.575), g.sub(1.0, g.ss(s, 0.925, 0.95)))

    def pin(c):
        return g.sub(1.0, g.ss(g.ab(g.sub(s, c)), 0.010, 0.022))
    pins = g.mx(pin(0.5625), pin(0.9375))
    dx = g.mul(g.sub(s, 0.75), period)
    dz = g.mul(g.sub(g.fr(g.add(g.mul(z, 1.0 / 0.09), 0.25)), 0.5), 0.09)
    diamond = g.add(g.div(g.ab(dx), 0.0105), g.div(g.ab(dz), 0.021))
    motif = g.mul(g.sub(1.0, g.ss(diamond, 0.78, 1.0)), lite)
    col = g.mixc(L["paper"], L["paper_light"], lite)
    col = g.mixc(col, L["paper_motif"], g.mul(motif, 0.75))
    col = g.mixc(col, L["paper_pin"], g.mul(pins, 0.9))
    co = g.texco("Object")
    blotch = g.noise(co, 1.3, detail=3.0, rough=0.6)                               # slow tonal drift
    col = g.vm("SCALE", col, None, None, g.mad(blotch, 0.22, 0.89))
    col = g.mixc(col, L["frieze"], g.ss(z, rail_z - 0.02, rail_z - 0.005))        # plain above the picture rail
    grain = g.noise(g.mapping(co, scale=(1.0, 1.0, 0.35)), 420.0, detail=1.0, rough=0.5)
    relief = g.sub(g.mad(grain, 0.6, g.mul(g.add(lite, motif), 0.5)), g.mul(pins, 0.7))
    b = principled(g, col, rough=0.84, spec=0.22, normal=g.bump(relief, 0.05, 0.0008))
    finish(g, b.outputs[0])
    return m


def mat_paint(R, name, color, rough=0.42, bump=0.012, scale=26.0):
    """Satin paint with faint brush marks."""
    m, g = R.mat(name)
    co = g.texco("Object")
    brush = g.noise(g.mapping(co, scale=(1.0, 1.0, 0.4)), scale, detail=2.0, rough=0.5)
    b = principled(g, color, rough=rough, spec=0.45, normal=g.bump(brush, bump, 0.001))
    finish(g, b.outputs[0])
    return m


def mat_plaster(R, name, color, scale=5.0, rough=0.9):
    """Matt plaster / ceiling paint: slow mottling and a fine grain."""
    m, g = R.mat(name)
    co = g.texco("Object")
    mott = g.noise(co, scale, detail=5.0, rough=0.6)
    grain = g.noise(co, 140.0, detail=1.0, rough=0.5)
    col = g.vm("SCALE", color, None, None, g.mad(mott, 0.16, 0.92))
    b = principled(g, col, rough=rough, spec=0.2, normal=g.bump(g.mad(grain, 0.6, g.mul(mott, 0.4)), 0.06, 0.002))
    finish(g, b.outputs[0])
    return m


def mat_plain(R, name, color, rough=0.9, metal=0.0, spec=0.5, **extra):
    m, g = R.mat(name)
    b = principled(g, color, rough=rough, metal=metal, spec=spec, **extra)
    finish(g, b.outputs[0])
    return m


# ---------------------------------------------------------------------------------------------------- floor


def mat_parquet(R, L, plank_len, plank_w):
    """Varnished parquet. UV map "plank" is (0..1 along, 0..1 across) per plank, "rand" two random numbers per plank:
    tone and brightness per plank, grain along the plank, the chamfer shading into a dark groove all round."""
    m, g = R.mat("Parquet")
    pu, pv, _ = g.sep(g.node("ShaderNodeUVMap", uv_map="plank").outputs[0])
    r1, r2, _ = g.sep(g.node("ShaderNodeUVMap", uv_map="rand").outputs[0])
    edge = g.mn(g.mn(g.mul(pu, plank_len), g.mul(g.inv(pu), plank_len)),
                g.mn(g.mul(pv, plank_w), g.mul(g.inv(pv), plank_w)))                      # metres to the nearest edge
    groove = g.sub(1.0, g.ss(edge, 0.0, 0.0009))             # dark at the foot of the 0.8 mm chamfer, gone at its top
    tone = g.mixc(L["floor_a"], L["floor_b"], r1)
    tone = g.vm("SCALE", tone, None, None, g.mad(r2, 0.30, 0.85))
    grain_co = g.xyz(g.mul(pu, plank_len * 6.0), g.mul(pv, plank_w * 70.0), g.mul(r2, 31.0))
    grain = g.noise(grain_co, 1.0, detail=5.0, rough=0.6, distortion=0.35)
    streak = g.ss(grain, 0.38, 0.72)
    col = g.mixc(tone, g.vm("SCALE", tone, None, None, 0.62), g.mul(streak, 0.7))
    col = g.mixc(col, L["floor_groove"], g.mul(groove, 0.85))
    relief = g.sub(g.mad(grain, 0.12, 0.0), g.mul(groove, 0.55))
    b = principled(g, col, rough=0.30, spec=0.55, normal=g.bump(relief, 0.35, 0.002))
    finish(g, b.outputs[0])
    return m


# ---------------------------------------------------------------------------------------------------- window


def mat_glass(R, L):
    """Night glass: clear, with a faint Fresnel reflection that smudges slightly."""
    m, g = R.mat("Glass", blend="DITHERED")
    fres = g.node("ShaderNodeFresnel")
    g.put(fres.inputs["IOR"], 1.45)
    smudge = g.noise(g.texco("Object"), 2.5, detail=4.0, rough=0.65)
    refl = g.add(g.mul(fres.outputs[0], 0.9), g.mad(smudge, 0.03, 0.015))
    gloss = g.node("ShaderNodeBsdfGlossy")
    g.put(gloss.inputs["Color"], (0.9, 0.95, 1.0, 1.0))
    g.put(gloss.inputs["Roughness"], 0.03)
    clear = g.node("ShaderNodeBsdfTransparent")
    g.put(clear.inputs["Color"], L["glass"])
    mix = g.node("ShaderNodeMixShader")
    g.put(mix.inputs[0], refl)
    g.nt.links.new(clear.outputs[0], mix.inputs[1])
    g.nt.links.new(gloss.outputs[0], mix.inputs[2])
    finish(g, mix.outputs[0])
    return m


def mat_slat(R, L):
    """Painted aluminium slats: satin, slightly metallic."""
    m, g = R.mat("BlindSlat")
    co = g.texco("Object")
    streak = g.noise(g.mapping(co, scale=(0.05, 40.0, 40.0)), 8.0, detail=2.0, rough=0.5)
    b = principled(g, g.vm("SCALE", L["slat"], None, None, g.mad(streak, 0.16, 0.92)), rough=0.36, metal=0.55, spec=0.6)
    finish(g, b.outputs[0])
    return m


# ---------------------------------------------------------------------------------------------------- neon, bulb


def mat_neon(R, L):
    """The neon tube: an emission whose strength is the set's `neon` parameter."""
    m, g = R.mat("Neon")
    em = g.node("ShaderNodeEmission")
    g.put(em.inputs["Color"], L["neon"])
    g.put(em.inputs["Strength"], g.mul(g.param("neon"), 9.0))
    core = principled(g, L["neon_tube"], rough=0.25, spec=0.5)
    mix = g.node("ShaderNodeMixShader")
    g.put(mix.inputs[0], g.ss(g.param("neon"), 0.0, 0.05))
    g.nt.links.new(core.outputs[0], mix.inputs[1])
    g.nt.links.new(em.outputs[0], mix.inputs[2])
    finish(g, mix.outputs[0])
    return m


def scale_emission(material, params, name="city_glow"):
    """Multiply the strength of every emission in a material's node tree whose strength is linked or is not 1.0 (the
    lit windows, the glow sheet, the beacons; the plain silhouettes at 1.0 keep their tone) by the output `name` of the
    set's parameter group `params`."""
    nt = material.node_tree
    grp = None
    for n in list(nt.nodes):
        if n.bl_idname == "ShaderNodeEmission":
            sock = n.inputs["Strength"]
        elif n.bl_idname == "ShaderNodeBsdfPrincipled":
            sock = n.inputs["Emission Strength"]
        else:
            continue
        if not sock.is_linked and (abs(sock.default_value - 1.0) < 1e-9 or sock.default_value == 0.0):
            continue
        if grp is None:
            grp = nt.nodes.new("ShaderNodeGroup")
            grp.node_tree = params
            grp.location = (n.location[0] - 400, n.location[1] - 400)
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.location = (n.location[0] - 160, n.location[1] - 200)
        if sock.is_linked:
            nt.links.new(sock.links[0].from_socket, mul.inputs[0])
        else:
            mul.inputs[0].default_value = sock.default_value
        nt.links.new(grp.outputs[name], mul.inputs[1])
        nt.links.new(mul.outputs[0], sock)


__all__ = ["params_group", "mat_wallpaper", "mat_paint", "mat_plaster", "mat_plain", "mat_parquet", "mat_glass",
           "mat_slat", "mat_neon", "scale_emission"]
