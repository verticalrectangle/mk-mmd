// The deferred lighting pass: on one sample, every pixel's closures (left in the G-buffer by the material passes,
// gbuffer.wgsl) lit by the scene's lights (light.wgsl), written with their emission as the sample's picture. The
// position is rebuilt from the depth (the sample's raster) plus the motion offset back to the frame's own time, where
// shading and shadows live. With EEVEE's ray tracing (frame.rt.x 1: raytrace.wgsl) the glossy lobes trace the screen,
// and the picture is kept as the light the next sample's rays find (radiance_out); imm.draw 1 is the pass before the
// first sample's, which only lights that picture for it, its lobes reading the probe.
//
// With many lights (pipelines.rs TILED_FROM) light_pass_tiled culls them per 8 x 8 tile first, by the box of the
// tile's surface points (light_reaches_box: influence radius, a one-sided area light's back, a spot's cone), a bit per
// light; its pixels run only those, in the order light_shading runs them all, skipping only lights that leave them
// unlit: the picture is the same to the bit. Lights past the first TILE_LIGHTS go to every pixel.

@group(1) @binding(0) var gb_base: texture_2d<u32>;
@group(1) @binding(1) var gb_motion: texture_2d<f32>;
@group(1) @binding(2) var gb_depth: texture_depth_2d;
@group(1) @binding(3) var<storage, read> gb_extras: array<vec4<u32>>;
@group(1) @binding(4) var color_out: texture_storage_2d<rgba16float, write>;

const TILE_LIGHTS: u32 = 1024u;
// the tile's box (its surface points' coordinates as tile_key orders them) and its lights (TILE_LIGHTS / 32 words)
var<workgroup> tile_lo: array<atomic<u32>, 3>;
var<workgroup> tile_hi: array<atomic<u32>, 3>;
var<workgroup> tile_mask: array<atomic<u32>, 32>;

// A float's bits in an order that compares as the floats do (for atomicMin / atomicMax), and back.
fn tile_key(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

fn tile_value(k: u32) -> f32 {
    return bitcast<f32>(select(~k, k & 0x7fffffffu, (k & 0x80000000u) != 0u));
}

// light_shading over the tile's lights.
fn light_shading_tile(h: Shading) -> vec3<f32> {
    var out = vec3<f32>(0.0);
    if (!takes_light(h)) {
        return out;
    }
    let n = min(frame.light_count, TILE_LIGHTS);
    for (var k = 0u; k * 32u < n; k = k + 1u) {
        var bits = atomicLoad(&tile_mask[k]);
        while (bits != 0u) {
            let i = k * 32u + firstTrailingBit(bits);
            bits = bits & (bits - 1u);
            out = light_add(out, i, h);
        }
    }
    for (var i = TILE_LIGHTS; i < frame.light_count; i = i + 1u) {
        out = light_add(out, i, h);
    }
    return out;
}

// A pixel without a surface: black (the world pass fills it later).
fn light_none(px: vec2<u32>) {
    textureStore(color_out, px, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    if (RT_TRACED && frame.rt.x > 0.5) {
        textureStore(radiance_out, px, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
    if (RT_TRACED && frame.rt.x > 1.5 && imm.draw == 0u) {
        textureStore(history_out, px, vec4<f32>(0.0));
    }
}

// A surface pixel's position on the frame's own time: from the sample's depth, plus the motion offset back.
fn light_position(px: vec2<u32>, sd: SampleData) -> vec3<f32> {
    let depth = textureLoad(gb_depth, px, 0);
    let ndc = vec2<f32>((f32(px.x) + 0.5) * frame.size.z * 2.0 - 1.0, 1.0 - (f32(px.y) + 0.5) * frame.size.w * 2.0);
    let ph = sd.vp_inv * vec4<f32>(ndc, depth, 1.0);
    var P = ph.xyz / ph.w;
    if ((frame.flags & 1u) != 0u) {
        P = P + textureLoad(gb_motion, px, 0).xyz;
    }
    return P;
}

// Every light on every pixel: what scenes of few lights draw (render.rs picks the entry point by the light count).
@compute @workgroup_size(8, 8)
fn light_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(frame.size.x);
    if (gid.x >= w || gid.y >= u32(frame.size.y)) {
        return;
    }
    let px = gid.xy;
    let base = textureLoad(gb_base, px, 0);
    if (((base.w >> 28u) & GB_SURFACE) == 0u) {
        light_none(px);
        return;
    }
    let sd = samples[imm.sample];
    light_pixel(px, base, light_position(px, sd), sd, false);
}

// The tile's lights alone on each pixel.
@compute @workgroup_size(8, 8)
fn light_pass_tiled(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let w = u32(frame.size.x);
    let inside = gid.x < w && gid.y < u32(frame.size.y);
    let px = gid.xy;
    let sd = samples[imm.sample];
    if (lid < 3u) {
        atomicStore(&tile_lo[lid], 0xffffffffu);
        atomicStore(&tile_hi[lid], 0u);
    }
    workgroupBarrier();
    var base = vec4<u32>(0u);
    var P = vec3<f32>(0.0);
    if (inside) {
        base = textureLoad(gb_base, px, 0);
        if (((base.w >> 28u) & GB_SURFACE) != 0u) {
            P = light_position(px, sd);
            for (var a = 0u; a < 3u; a = a + 1u) {
                atomicMin(&tile_lo[a], tile_key(P[a]));
                atomicMax(&tile_hi[a], tile_key(P[a]));
            }
        }
    }
    workgroupBarrier();
    // the lights that may reach the tile's surfaces, a light per invocation at a time
    let klo = vec3<u32>(atomicLoad(&tile_lo[0]), atomicLoad(&tile_lo[1]), atomicLoad(&tile_lo[2]));
    let khi = vec3<u32>(atomicLoad(&tile_hi[0]), atomicLoad(&tile_hi[1]), atomicLoad(&tile_hi[2]));
    if (klo.x <= khi.x) {
        let lo = vec3<f32>(tile_value(klo.x), tile_value(klo.y), tile_value(klo.z));
        let hi = vec3<f32>(tile_value(khi.x), tile_value(khi.y), tile_value(khi.z));
        let n = min(frame.light_count, TILE_LIGHTS);
        for (var i = lid; i < n; i = i + 64u) {
            if (light_reaches_box(lights[i], lo, hi)) {
                atomicOr(&tile_mask[i / 32u], 1u << (i % 32u));
            }
        }
    }
    workgroupBarrier();
    if (!inside) {
        return;
    }
    if (((base.w >> 28u) & GB_SURFACE) == 0u) {
        light_none(px);
        return;
    }
    light_pixel(px, base, P, sd, true);
}

// A surface pixel's closures (G-buffer `base` and extras) at P lit by every light or (`tiled`) the tile's, its world
// light or traced light added, written out.
fn light_pixel(px: vec2<u32>, base: vec4<u32>, P: vec3<f32>, sd: SampleData, tiled: bool) {
    let w = u32(frame.size.x);
    let flags = base.w >> 28u;
    g_pixel = px;
    g_sample = sd.info.y;
    g_unshadowed = (flags & GB_UNSHADOWED) != 0u;
    var h: Shading;
    h.P = P;
    h.V = view_dir(P);
    h.Ng = gb_unpack_n8(base.w & 0xffffu);
    h.diffuse = gb_from_rgb9e5(base.x);
    h.diffuse_N = gb_unpack_n16(base.z);
    h.reflection = gb_from_rgb9e5(base.y);
    h.reflection_N = h.diffuse_N;
    h.translucent = vec3<f32>(0.0);
    h.translucent_N = -h.diffuse_N;
    h.coat = vec3<f32>(0.0);
    h.coat_N = h.diffuse_N;
    let rough = gb_unpack_rough12((base.w >> 16u) & 4095u);
    var coat_rough = 0.0;
    var emission = vec3<f32>(0.0);
    h.translucent_slab = select(0.0, 1.0, (flags & GB_SLAB) != 0u);
    if ((flags & GB_EXTRAS) != 0u) {
        let i = (px.y * w + px.x) * 2u;
        let e0 = gb_extras[i];
        let e1 = gb_extras[i + 1u];
        emission = gb_from_rgb9e5(e0.x);
        h.translucent = gb_from_rgb9e5(e0.y);
        h.translucent_N = -gb_unpack_n16(e0.z);
        h.reflection_N = gb_unpack_n16(e0.w);
        h.coat = gb_from_rgb9e5(e1.x);
        h.coat_N = gb_unpack_n16(e1.y);
        let q = unpack2x16unorm(e1.z).x;
        coat_rough = q * q;
    }
    h = shading_finish(h, rough, coat_rough);
    var lit = vec3<f32>(0.0);
    if (tiled) {
        lit = light_shading_tile(h);
    } else {
        lit = light_shading(h);
    }
    var c = emission + lit;
    // what the next sample's rays find: the raw traced light (EEVEE's denoiser filters what the film takes)
    var fed = c;
    if (RT_TRACED && frame.rt.x > 0.5 && imm.draw == 0u) {
        c = c + traced_light(h, rough, coat_rough);
        fed = c;
        if (frame.rt.x > 1.5) {
            c = c - rt_reflected + rt_temporal(rt_reflected);
        }
    } else if (frame.world.x > 0.5) {
        c = c + world_light(h, rough, coat_rough);
        fed = c;
    }
    textureStore(color_out, px, vec4<f32>(c, 1.0));
    if (RT_TRACED && frame.rt.x > 0.5) {
        // the next sample's rays read the light without fast GI's bounce (raytrace.wgsl rt_bounce: zero unless traced)
        textureStore(radiance_out, px, vec4<f32>(fed - rt_bounce, 1.0));
    }
}
