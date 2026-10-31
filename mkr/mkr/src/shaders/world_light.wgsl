// The world's light on a point's closures, as EEVEE Next lights with its world when it traces no rays (lab world_*:
// no occlusion of it at all): diffuse and translucent closures from the L1 spherical harmonics of the world's probe
// (world_probe.wgsl), its band 1 weighted as EEVEE's frames show (frame.world.w of the cosine lobe's 2/3), glossy lobes
// from the probe along the reflection at a level by roughness, the roughest blending into the harmonics' light. Off
// (frame.world.x 0) without a world.

// The light a Lambert lobe facing N gets from the world, over pi (E(N) / pi).
fn world_irradiance(N: vec3<f32>) -> vec3<f32> {
    let b1 = 0.48860251 * (N.y * world_sh[1].rgb + N.z * world_sh[2].rgb + N.x * world_sh[3].rgb);
    return max(frame.world.z * 0.28209479 * world_sh[0].rgb + frame.world.w * (2.0 / 3.0) * b1, vec3<f32>(0.0));
}

// The probe's radiance around direction R blurred as a glossy lobe of `roughness` blurs it: the box-filtered mip whose
// texels are frame.world_probe.x lobe widths (GGX alpha = roughness²) across, frame.world.y its last mip; lobes
// from frame.world_probe.y to .z roughness blend into the harmonics' light around R.
fn world_glossy(R: vec3<f32>, roughness: f32) -> vec3<f32> {
    let alpha = max(roughness * roughness, 1e-4);
    let lod = clamp(log2(frame.world_probe.x * alpha * frame.world_probe.w / 1.5707963), 0.0, frame.world.y);
    let probe = textureSampleLevel(world_probe, s_linear_extend, R, lod).rgb;
    let t = smoothstep(frame.world_probe.y, frame.world_probe.z, roughness);
    if (t <= 0.0) {
        return probe;
    }
    return mix(probe, world_irradiance(R), t);
}

fn world_light(h: Shading, reflection_roughness: f32, coat_roughness: f32) -> vec3<f32> {
    var o = vec3<f32>(0.0);
    if (any(h.diffuse > vec3<f32>(0.0))) {
        o = o + h.diffuse * world_irradiance(h.diffuse_N);
    }
    if (any(h.translucent > vec3<f32>(0.0))) {
        // as lamps light it (light.wgsl): the closure colour squared; a sphere's (SPHERE) from every side alike, the
        // world's mean radiance; a slab's as Lambert around the side its light comes from
        var t = 0.28209479 * world_sh[0].rgb * frame.world.z;
        if (h.translucent_slab > 0.5) {
            t = world_irradiance(h.translucent_N);
        }
        o = o + h.translucent * h.translucent * t * frame.scale.x;
    }
    if (any(h.reflection > vec3<f32>(0.0))) {
        o = o + h.reflection * world_glossy(reflect(-h.V, h.reflection_N), reflection_roughness);
    }
    if (any(h.coat > vec3<f32>(0.0))) {
        o = o + h.coat * world_glossy(reflect(-h.V, h.coat_N), coat_roughness);
    }
    return o;
}
