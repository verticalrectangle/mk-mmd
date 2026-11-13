// The material side of the G-buffer (gbuffer.wgsl): a material module's fg_ entry evaluates its closures and leaves
// them here for the lighting pass. Group 1 of the G-buffer pass: the extras, one pair of vec4<u32> per pixel, written
// only by the surface the depth prepass kept (the entry forces early depth tests, so hidden fragments never run).

@group(1) @binding(0) var<storage, read_write> gb_extras: array<vec4<u32>>;

fn gbuffer_write(c_in: Closures, s: Surface, in: VsOut) -> GOut {
    // DITHERED: the prepass kept this surface with the probability of its opacity, so its closures count over its
    // coverage
    let a = mks_opacity(c_in);
    var c = c_in;
    if (a > 0.0 && a < 1.0) {
        c = mks_closures_scale(c_in, 1.0 / a);
    }
    let has_diffuse = any(c.diffuse > vec3<f32>(0.0));
    let has_refl = any(c.reflection > vec3<f32>(0.0));
    let N = to_gltf(select(c.reflection_N, c.diffuse_N, has_diffuse || !has_refl));
    let refl_N = to_gltf(c.reflection_N);
    let own_refl_N = has_refl && has_diffuse && any(abs(refl_N - N) > vec3<f32>(1e-5));
    let has_extras = any(c.emission > vec3<f32>(0.0)) || any(c.translucent > vec3<f32>(0.0))
                  || any(c.coat > vec3<f32>(0.0)) || own_refl_N;
    let flags = GB_SURFACE | select(0u, GB_EXTRAS, has_extras) | select(0u, GB_SLAB, c.translucent_slab > 0.5)
              | select(0u, GB_UNSHADOWED, g_unshadowed);
    var o: GOut;
    o.base = vec4<u32>(gb_rgb9e5(c.diffuse), gb_rgb9e5(c.reflection), gb_pack_n16(N),
                       gb_pack_n8(to_gltf(s.Ng)) | (gb_pack_rough12(c.reflection_roughness) << 16u) | (flags << 28u));
    o.motion = vec4<f32>(in.p - in.ps, 0.0);
    if ((frame.flags & 2u) != 0u) {
        o.motion = mb_velocity(in);
    }
    if (has_extras) {
        let px = vec2<u32>(in.pos.xy);
        let i = (px.y * u32(frame.size.x) + px.x) * 2u;
        gb_extras[i] = vec4<u32>(gb_rgb9e5(c.emission), gb_rgb9e5(c.translucent), gb_pack_n16(to_gltf(c.translucent_N)),
                                 gb_pack_n16(refl_N));
        gb_extras[i + 1u] = vec4<u32>(gb_rgb9e5(c.coat), gb_pack_n16(to_gltf(c.coat_N)),
                                      pack2x16unorm(vec2<f32>(sqrt(clamp(c.coat_roughness, 0.0, 1.0)), 0.0)), 0u);
    }
    return o;
}
